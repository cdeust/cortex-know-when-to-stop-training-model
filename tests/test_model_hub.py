"""Tests for model_hub.download_model (urlretrieve is faked; no network)."""

from pathlib import Path

from cortex_beam_abstain import model_hub


def _fake_retrieve(calls):
    def retrieve(url, dest):
        calls.append(url)
        Path(dest).write_bytes(b"data")

    return retrieve


def test_downloads_both_files(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(model_hub.urllib.request, "urlretrieve", _fake_retrieve(calls))
    result = model_hub.download_model("org/m", cache_dir=tmp_path / "c")
    assert result == tmp_path / "c" / "model.onnx"
    assert (tmp_path / "c" / "tokenizer.json").exists()
    assert [u.rsplit("/", 1)[1] for u in calls] == ["model.onnx", "tokenizer.json"]


def test_existing_model_short_circuits(tmp_path, monkeypatch):
    (tmp_path / "model.onnx").write_bytes(b"x")
    calls = []
    monkeypatch.setattr(model_hub.urllib.request, "urlretrieve", _fake_retrieve(calls))
    assert model_hub.download_model(cache_dir=tmp_path) == tmp_path / "model.onnx"
    assert calls == []


def test_existing_tokenizer_is_skipped(tmp_path, monkeypatch):
    (tmp_path / "tokenizer.json").write_text("{}")
    calls = []
    monkeypatch.setattr(model_hub.urllib.request, "urlretrieve", _fake_retrieve(calls))
    model_hub.download_model(cache_dir=tmp_path)
    assert len(calls) == 1 and calls[0].endswith("model.onnx")


def test_failure_cleans_partial_and_returns_none(tmp_path, monkeypatch):
    def retrieve(url, dest):
        Path(dest).write_bytes(b"partial")
        raise OSError("network down")

    monkeypatch.setattr(model_hub.urllib.request, "urlretrieve", retrieve)
    assert model_hub.download_model(cache_dir=tmp_path) is None
    assert not (tmp_path / "model.onnx").exists()


def test_failure_without_partial_file(tmp_path, monkeypatch):
    def retrieve(url, dest):
        raise OSError("refused")

    monkeypatch.setattr(model_hub.urllib.request, "urlretrieve", retrieve)
    assert model_hub.download_model(cache_dir=tmp_path) is None


def test_default_cache_dir_under_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(model_hub.urllib.request, "urlretrieve", _fake_retrieve([]))
    result = model_hub.download_model()
    assert result == tmp_path / ".cache" / "cortex-beam-abstain" / "model.onnx"
