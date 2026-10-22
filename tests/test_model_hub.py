"""Behaviour tests for model download and caching."""

from pathlib import Path

from cortex_beam_abstain import model_hub


def fake_retrieve_factory(fail_on=None, calls=None):
    def retrieve(url, dest):
        if calls is not None:
            calls.append(url)
        if fail_on and url.endswith(fail_on):
            Path(dest).write_bytes(b"partial")
            raise OSError("network down")
        Path(dest).write_bytes(b"data")

    return retrieve


def test_downloads_both_files(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(model_hub.urllib.request, "urlretrieve", fake_retrieve_factory(calls=calls))
    result = model_hub.download_model("org/model", cache_dir=tmp_path / "c")
    assert result == tmp_path / "c" / "model.onnx"
    assert (tmp_path / "c" / "tokenizer.json").exists()
    assert calls == [
        "https://huggingface.co/org/model/resolve/main/model.onnx",
        "https://huggingface.co/org/model/resolve/main/tokenizer.json",
    ]


def test_cached_model_skips_download(monkeypatch, tmp_path):
    (tmp_path / "model.onnx").write_bytes(b"x")
    calls = []
    monkeypatch.setattr(model_hub.urllib.request, "urlretrieve", fake_retrieve_factory(calls=calls))
    assert model_hub.download_model(cache_dir=tmp_path) == tmp_path / "model.onnx"
    assert calls == []


def test_existing_tokenizer_not_redownloaded(monkeypatch, tmp_path):
    (tmp_path / "tokenizer.json").write_text("{}")
    calls = []
    monkeypatch.setattr(model_hub.urllib.request, "urlretrieve", fake_retrieve_factory(calls=calls))
    assert model_hub.download_model(cache_dir=tmp_path) == tmp_path / "model.onnx"
    assert len(calls) == 1 and calls[0].endswith("model.onnx")


def test_failure_cleans_partial_download(monkeypatch, tmp_path):
    monkeypatch.setattr(
        model_hub.urllib.request, "urlretrieve", fake_retrieve_factory(fail_on="tokenizer.json")
    )
    assert model_hub.download_model(cache_dir=tmp_path) is None
    assert not (tmp_path / "tokenizer.json").exists()


def test_failure_without_partial_file(monkeypatch, tmp_path):
    def retrieve(url, dest):
        raise OSError("nope")

    monkeypatch.setattr(model_hub.urllib.request, "urlretrieve", retrieve)
    assert model_hub.download_model(cache_dir=tmp_path) is None


def test_default_cache_dir_is_home(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(model_hub.urllib.request, "urlretrieve", fake_retrieve_factory())
    result = model_hub.download_model()
    assert result == tmp_path / ".cache" / "cortex-beam-abstain" / "model.onnx"
