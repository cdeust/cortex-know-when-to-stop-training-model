"""Tests for model resolution, ONNX loading and inference (all faked, offline)."""

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from cortex_beam_abstain import classifier as clf_mod
from cortex_beam_abstain.classifier import AbstentionClassifier


class FakeEncoded:
    def __init__(self, n=3):
        self.ids = list(range(1, n + 1))
        self.attention_mask = [1] * n
        self.type_ids = [0] * n


class FakeTokenizer:
    loaded_from = None

    @classmethod
    def from_file(cls, path):
        cls.loaded_from = ("file", path)
        return cls()

    @classmethod
    def from_pretrained(cls, name):
        cls.loaded_from = ("pretrained", name)
        return cls()

    def encode(self, query, passage):
        return FakeEncoded()


class FakeInput:
    def __init__(self, name):
        self.name = name


class FakeSession:
    input_names = ["input_ids", "attention_mask"]
    logits = [0.0, 2.0]

    def __init__(self, path, providers=None):
        self.path = path
        self.last_inputs = None

    def get_inputs(self):
        return [FakeInput(n) for n in self.input_names]

    def run(self, _outputs, inputs):
        self.last_inputs = inputs
        return [np.array([self.logits])]


@pytest.fixture
def fake_onnx(monkeypatch):
    ort = types.ModuleType("onnxruntime")
    ort.InferenceSession = FakeSession
    tok = types.ModuleType("tokenizers")
    tok.Tokenizer = FakeTokenizer
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    monkeypatch.setitem(sys.modules, "tokenizers", tok)
    FakeTokenizer.loaded_from = None
    FakeSession.input_names = ["input_ids", "attention_mask"]


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    return tmp_path


class TestResolveAndFallback:
    def test_missing_explicit_path_falls_back_to_heuristic(self, tmp_path):
        clf = AbstentionClassifier(model_path=tmp_path / "nope.onnx")
        assert clf._use_heuristic is True
        assert 0.0 <= clf.predict("a b", "a b c") <= 1.0

    def test_local_cache_is_used(self, home, fake_onnx):
        cache = home / ".cache" / "cortex-beam-abstain"
        cache.mkdir(parents=True)
        (cache / "model.onnx").write_bytes(b"x")
        clf = AbstentionClassifier()
        assert clf._use_heuristic is False
        assert clf._session.path == str(cache / "model.onnx")

    def test_auto_download_success(self, home, fake_onnx, monkeypatch):
        model = home / "dl" / "model.onnx"
        model.parent.mkdir()
        model.write_bytes(b"x")
        monkeypatch.setattr(
            "cortex_beam_abstain.model_hub.download_model",
            lambda model_id, cache_dir=None: model,
        )
        clf = AbstentionClassifier()
        assert clf._use_heuristic is False

    def test_auto_download_none_falls_back(self, home, monkeypatch):
        monkeypatch.setattr(
            "cortex_beam_abstain.model_hub.download_model",
            lambda model_id, cache_dir=None: None,
        )
        assert AbstentionClassifier()._use_heuristic is True

    def test_auto_download_exception_falls_back(self, home, monkeypatch):
        def boom(model_id, cache_dir=None):
            raise RuntimeError("offline")

        monkeypatch.setattr("cortex_beam_abstain.model_hub.download_model", boom)
        assert AbstentionClassifier()._use_heuristic is True


class TestLoadOnnx:
    def test_tokenizer_next_to_model(self, tmp_path, fake_onnx):
        (tmp_path / "model.onnx").write_bytes(b"x")
        (tmp_path / "tokenizer.json").write_text("{}")
        AbstentionClassifier(model_path=tmp_path / "model.onnx")
        assert FakeTokenizer.loaded_from[0] == "file"

    def test_tokenizer_pretrained_fallback(self, tmp_path, fake_onnx):
        (tmp_path / "model.onnx").write_bytes(b"x")
        AbstentionClassifier(model_path=tmp_path / "model.onnx")
        assert FakeTokenizer.loaded_from == ("pretrained", "distilbert-base-uncased")

    def test_import_error_falls_back(self, tmp_path, monkeypatch):
        (tmp_path / "model.onnx").write_bytes(b"x")
        monkeypatch.setitem(sys.modules, "onnxruntime", None)
        clf = AbstentionClassifier(model_path=tmp_path / "model.onnx")
        assert clf._use_heuristic is True

    def test_load_error_falls_back(self, tmp_path, fake_onnx, monkeypatch):
        (tmp_path / "model.onnx").write_bytes(b"x")

        def bad(self, path, providers=None):
            raise ValueError("corrupt")

        monkeypatch.setattr(FakeSession, "__init__", bad)
        clf = AbstentionClassifier(model_path=tmp_path / "model.onnx")
        assert clf._use_heuristic is True


class TestPredictOnnx:
    @pytest.fixture
    def clf(self, tmp_path, fake_onnx):
        (tmp_path / "model.onnx").write_bytes(b"x")
        return AbstentionClassifier(model_path=tmp_path / "model.onnx")

    def test_softmax_probability(self, clf):
        expected = float(np.exp(2.0) / (1.0 + np.exp(2.0)))
        assert clf.predict("q", "p") == pytest.approx(expected)

    def test_inputs_padded_without_token_type_ids(self, clf):
        clf.predict("q", "p")
        inputs = clf._session.last_inputs
        assert set(inputs) == {"input_ids", "attention_mask"}
        assert inputs["input_ids"].shape == (1, clf_mod._MAX_LENGTH)
        assert inputs["attention_mask"].sum() == 3

    def test_token_type_ids_when_expected(self, clf):
        FakeSession.input_names = ["input_ids", "attention_mask", "token_type_ids"]
        clf.predict("q", "p")
        assert clf._session.last_inputs["token_type_ids"].shape == (
            1,
            clf_mod._MAX_LENGTH,
        )

    def test_missing_session_returns_neutral(self, clf):
        clf._session = None
        assert clf.predict("q", "p") == 0.5


class TestShouldAbstain:
    def test_all_below_threshold(self):
        clf = AbstentionClassifier(use_heuristic=True)
        assert clf.should_abstain("recipe cooking", ["bug api middleware"]) is True

    def test_relevant_passage_prevents_abstain(self):
        clf = AbstentionClassifier(use_heuristic=True)
        assert clf.should_abstain(
            "typescript frontend", ["typescript frontend project"], threshold=0.1
        ) is False

    def test_default_threshold_used(self):
        clf = AbstentionClassifier(use_heuristic=True, threshold=1.1)
        assert clf.should_abstain("same words", ["same words"]) is True
