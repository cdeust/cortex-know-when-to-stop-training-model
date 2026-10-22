"""Tests for the ONNX path, model resolution and loading of the classifier."""

import sys
import types
from pathlib import Path
from unittest import mock

import numpy as np

from cortex_beam_abstain import classifier as clf_mod
from cortex_beam_abstain.classifier import AbstentionClassifier


class FakeEncoding:
    def __init__(self, n=5):
        self.ids = list(range(1, n + 1))
        self.attention_mask = [1] * n
        self.type_ids = [0] * n


class FakeTokenizer:
    @classmethod
    def from_file(cls, path):
        return cls()

    @classmethod
    def from_pretrained(cls, name):
        return cls()

    def encode(self, query, passage):
        return FakeEncoding()


class FakeSession:
    def __init__(
        self, path=None, providers=None, input_names=("input_ids", "attention_mask")
    ):
        self.input_names = input_names
        self.last_inputs = None

    def get_inputs(self):
        return [types.SimpleNamespace(name=n) for n in self.input_names]

    def run(self, _outputs, inputs):
        self.last_inputs = inputs
        return [np.array([[0.0, 2.0]])]


def _fake_modules(session_cls=FakeSession):
    ort = types.ModuleType("onnxruntime")
    ort.InferenceSession = session_cls
    tok = types.ModuleType("tokenizers")
    tok.Tokenizer = FakeTokenizer
    return {"onnxruntime": ort, "tokenizers": tok}


def _loaded(tmp_path, with_tokenizer=True, session_cls=FakeSession):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"x")
    if with_tokenizer:
        (tmp_path / "tokenizer.json").write_text("{}")
    with mock.patch.dict(sys.modules, _fake_modules(session_cls)):
        return AbstentionClassifier(model_path=model)


class TestLoading:
    def test_loads_with_tokenizer_file(self, tmp_path):
        clf = _loaded(tmp_path)
        assert clf._use_heuristic is False
        assert clf._session is not None

    def test_loads_with_pretrained_tokenizer(self, tmp_path):
        clf = _loaded(tmp_path, with_tokenizer=False)
        assert clf._tokenizer is not None

    def test_missing_runtime_falls_back(self, tmp_path):
        model = tmp_path / "model.onnx"
        model.write_bytes(b"x")
        with mock.patch.dict(sys.modules, {"onnxruntime": None}):
            clf = AbstentionClassifier(model_path=model)
        assert clf._use_heuristic is True

    def test_load_error_falls_back(self, tmp_path):
        def boom(*a, **k):
            raise RuntimeError("bad model")

        clf = _loaded(tmp_path, session_cls=boom)
        assert clf._use_heuristic is True

    def test_missing_model_path_falls_back(self, tmp_path):
        clf = AbstentionClassifier(model_path=tmp_path / "absent.onnx")
        assert clf._use_heuristic is True
        assert 0.0 <= clf.predict("a b", "a b c") <= 1.0


class TestResolve:
    def test_local_cache_hit(self, tmp_path):
        cache = tmp_path / ".cache" / "cortex-beam-abstain"
        cache.mkdir(parents=True)
        (cache / "model.onnx").write_bytes(b"x")
        clf = AbstentionClassifier(use_heuristic=True)
        with mock.patch.object(Path, "home", return_value=tmp_path):
            assert clf._resolve_model(None) == cache / "model.onnx"

    def test_download_used_when_no_cache(self, tmp_path):
        clf = AbstentionClassifier(use_heuristic=True)
        target = tmp_path / "dl.onnx"
        with mock.patch.object(Path, "home", return_value=tmp_path), mock.patch(
            "cortex_beam_abstain.model_hub.download_model", return_value=target
        ):
            assert clf._resolve_model(None) == target

    def test_download_failure_returns_none(self, tmp_path):
        clf = AbstentionClassifier(use_heuristic=True)
        with mock.patch.object(Path, "home", return_value=tmp_path), mock.patch(
            "cortex_beam_abstain.model_hub.download_model", side_effect=OSError("net")
        ):
            assert clf._resolve_model(None) is None

    def test_default_model_without_anything_uses_heuristic(self, tmp_path):
        with mock.patch.object(Path, "home", return_value=tmp_path), mock.patch(
            "cortex_beam_abstain.model_hub.download_model", return_value=None
        ):
            clf = AbstentionClassifier()
        assert clf._use_heuristic is True


class TestPredictOnnx:
    def test_probability_is_softmax(self, tmp_path):
        clf = _loaded(tmp_path)
        expected = float(np.exp(2.0) / (1.0 + np.exp(2.0)))
        assert abs(clf.predict("q", "p") - expected) < 1e-6

    def test_inputs_padded_to_max_length(self, tmp_path):
        clf = _loaded(tmp_path)
        clf.predict("q", "p")
        inputs = clf._session.last_inputs
        assert inputs["input_ids"].shape == (1, clf_mod._MAX_LENGTH)
        assert "token_type_ids" not in inputs

    def test_token_type_ids_when_expected(self, tmp_path):
        def factory(path=None, providers=None):
            return FakeSession(
                path, providers, ("input_ids", "attention_mask", "token_type_ids")
            )

        clf = _loaded(tmp_path, session_cls=factory)
        clf.predict("q", "p")
        shape = clf._session.last_inputs["token_type_ids"].shape
        assert shape == (1, clf_mod._MAX_LENGTH)

    def test_no_session_returns_neutral(self):
        clf = AbstentionClassifier(use_heuristic=True)
        assert clf._predict_onnx("q", "p") == 0.5


class TestBatchAndAbstain:
    def test_predict_batch(self):
        clf = AbstentionClassifier(use_heuristic=True)
        scores = clf.predict_batch([("a b", "a b"), ("x", "y")])
        assert len(scores) == 2

    def test_should_abstain_threshold_override(self):
        clf = AbstentionClassifier(use_heuristic=True)
        assert clf.should_abstain("a b c", ["a b c"], threshold=2.0) is True
        assert clf.should_abstain("a b c", ["a b c"], threshold=0.0) is False
