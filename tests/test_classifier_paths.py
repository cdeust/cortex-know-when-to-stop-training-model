"""Behaviour tests for model resolution, ONNX loading and inference paths."""

import logging
import sys
import types
from pathlib import Path

import numpy as np
import pytest

import cortex_beam_abstain
from cortex_beam_abstain import classifier as clf_mod
from cortex_beam_abstain.classifier import AbstentionClassifier


class FakeInput:
    def __init__(self, name):
        self.name = name


class FakeSession:
    input_names = ("input_ids", "attention_mask")
    logits = (0.0, 2.0)

    def __init__(self, path=None, providers=None):
        self.path = path
        self.providers = providers
        self.fed = None

    def get_inputs(self):
        return [FakeInput(n) for n in self.input_names]

    def run(self, _outputs, feed):
        self.fed = feed
        return [np.array([self.logits], dtype=np.float32)]


class TypeIdSession(FakeSession):
    input_names = ("input_ids", "attention_mask", "token_type_ids")


class FakeEncoding:
    def __init__(self, n):
        self.ids = list(range(1, n + 1))
        self.attention_mask = [1] * n
        self.type_ids = [1] * n


class FakeTokenizer:
    def __init__(self, n=5):
        self.n = n
        self.source = None

    def encode(self, query, passage):
        return FakeEncoding(self.n)

    @classmethod
    def from_file(cls, path):
        tok = cls()
        tok.source = ("file", path)
        return tok

    @classmethod
    def from_pretrained(cls, name):
        tok = cls()
        tok.source = ("pretrained", name)
        return tok


def install_fakes(monkeypatch, session_cls=FakeSession):
    ort = types.ModuleType("onnxruntime")
    ort.InferenceSession = session_cls
    tokenizers = types.ModuleType("tokenizers")
    tokenizers.Tokenizer = FakeTokenizer
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    monkeypatch.setitem(sys.modules, "tokenizers", tokenizers)


def make_model(tmp_path, tokenizer=True):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"x")
    if tokenizer:
        (tmp_path / "tokenizer.json").write_text("{}")
    return model


def test_package_exports():
    assert cortex_beam_abstain.__version__ == "0.1.0"
    assert set(cortex_beam_abstain.__all__) == {"AbstentionClassifier", "cosine_gap_score"}


def test_loads_model_with_local_tokenizer(monkeypatch, tmp_path):
    install_fakes(monkeypatch)
    clf = AbstentionClassifier(model_path=make_model(tmp_path))
    assert clf._use_heuristic is False
    assert clf._session.providers == ["CPUExecutionProvider"]
    assert clf._tokenizer.source[0] == "file"


def test_loads_model_without_tokenizer_uses_pretrained(monkeypatch, tmp_path):
    install_fakes(monkeypatch)
    clf = AbstentionClassifier(model_path=make_model(tmp_path, tokenizer=False))
    assert clf._tokenizer.source == ("pretrained", "distilbert-base-uncased")


def test_missing_explicit_model_falls_back_to_heuristic(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        clf = AbstentionClassifier(model_path=tmp_path / "nope.onnx")
    assert clf._use_heuristic is True
    assert "falling back to heuristic" in caplog.text
    assert 0.0 <= clf.predict("recipe cake", "cake recipe") <= 1.0


def test_missing_runtime_falls_back(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "onnxruntime", None)  # import raises ImportError
    clf = AbstentionClassifier(model_path=make_model(tmp_path))
    assert clf._use_heuristic is True


def test_load_failure_falls_back(monkeypatch, tmp_path):
    def boom(*args, **kwargs):
        raise RuntimeError("corrupt")

    install_fakes(monkeypatch, session_cls=boom)
    clf = AbstentionClassifier(model_path=make_model(tmp_path))
    assert clf._use_heuristic is True


def test_resolve_uses_local_cache(monkeypatch, tmp_path):
    cache = tmp_path / ".cache" / "cortex-beam-abstain"
    cache.mkdir(parents=True)
    (cache / "model.onnx").write_bytes(b"x")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    clf = AbstentionClassifier(use_heuristic=True)
    assert clf._resolve_model(None) == cache / "model.onnx"


def test_resolve_downloads_when_not_cached(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    calls = []

    def fake_download(model_id, cache_dir):
        calls.append((model_id, cache_dir))
        return tmp_path / "dl.onnx"

    monkeypatch.setattr("cortex_beam_abstain.model_hub.download_model", fake_download)
    clf = AbstentionClassifier(use_heuristic=True)
    assert clf._resolve_model(None) == tmp_path / "dl.onnx"
    assert calls[0][0] == "cdeust/cortex-beam-abstain-v1"


def test_resolve_download_error_returns_none(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    def fail(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr("cortex_beam_abstain.model_hub.download_model", fail)
    assert AbstentionClassifier(use_heuristic=True)._resolve_model(None) is None


def test_default_construction_without_model_uses_heuristic(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("cortex_beam_abstain.model_hub.download_model", lambda *a, **k: None)
    assert AbstentionClassifier()._use_heuristic is True


def test_predict_onnx_probability_and_padding(monkeypatch, tmp_path):
    install_fakes(monkeypatch)
    clf = AbstentionClassifier(model_path=make_model(tmp_path))
    score = clf.predict("q", "p")
    assert score == pytest.approx(np.exp(2.0) / (1.0 + np.exp(2.0)))
    fed = clf._session.fed
    assert fed["input_ids"].shape == (1, clf_mod._MAX_LENGTH)
    assert fed["attention_mask"].sum() == 5
    assert "token_type_ids" not in fed


def test_predict_onnx_truncates_and_sends_token_type_ids(monkeypatch, tmp_path):
    install_fakes(monkeypatch, session_cls=TypeIdSession)
    clf = AbstentionClassifier(model_path=make_model(tmp_path))
    clf._tokenizer = FakeTokenizer(n=clf_mod._MAX_LENGTH + 40)
    clf.predict("q", "p")
    fed = clf._session.fed
    assert fed["input_ids"].shape == (1, clf_mod._MAX_LENGTH)
    assert fed["token_type_ids"].shape == (1, clf_mod._MAX_LENGTH)
    assert fed["attention_mask"].sum() == clf_mod._MAX_LENGTH


def test_predict_onnx_without_session_is_neutral():
    clf = AbstentionClassifier(use_heuristic=True)
    assert clf._predict_onnx("q", "p") == 0.5


def test_should_abstain_threshold_behaviour():
    clf = AbstentionClassifier(use_heuristic=True)
    irrelevant = ["completely unrelated words"]
    relevant = ["chocolate cake recipe"]
    query = "chocolate cake recipe"
    assert clf.should_abstain(query, irrelevant) is True
    assert clf.should_abstain(query, irrelevant + relevant) is False
    assert clf.should_abstain(query, relevant, threshold=1.1) is True
    assert clf.should_abstain(query, irrelevant, threshold=0.0) is False
