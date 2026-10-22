"""Tests for the HuggingFace download helper (network is mocked)."""

from unittest import mock

from cortex_beam_abstain.model_hub import download_model


def _fake_retrieve(url, path):
    with open(path, "wb") as fh:
        fh.write(b"data")


class TestDownloadModel:
    def test_downloads_both_files(self, tmp_path):
        with mock.patch("urllib.request.urlretrieve", side_effect=_fake_retrieve):
            result = download_model("org/model", cache_dir=tmp_path)
        assert result == tmp_path / "model.onnx"
        assert (tmp_path / "tokenizer.json").exists()

    def test_existing_model_short_circuits(self, tmp_path):
        (tmp_path / "model.onnx").write_bytes(b"x")
        with mock.patch("urllib.request.urlretrieve") as retrieve:
            assert download_model(cache_dir=tmp_path) == tmp_path / "model.onnx"
        retrieve.assert_not_called()

    def test_existing_tokenizer_skipped(self, tmp_path):
        (tmp_path / "tokenizer.json").write_text("{}")
        with mock.patch(
            "urllib.request.urlretrieve", side_effect=_fake_retrieve
        ) as retrieve:
            download_model(cache_dir=tmp_path)
        assert retrieve.call_count == 1

    def test_failure_cleans_partial_and_returns_none(self, tmp_path):
        def partial(url, path):
            with open(path, "wb") as fh:
                fh.write(b"par")
            raise OSError("cut")

        with mock.patch("urllib.request.urlretrieve", side_effect=partial):
            assert download_model(cache_dir=tmp_path) is None
        assert not (tmp_path / "model.onnx").exists()

    def test_failure_without_partial_file(self, tmp_path):
        with mock.patch("urllib.request.urlretrieve", side_effect=OSError("no")):
            assert download_model(cache_dir=tmp_path) is None

    def test_default_cache_dir(self, tmp_path):
        with mock.patch("pathlib.Path.home", return_value=tmp_path), mock.patch(
            "urllib.request.urlretrieve", side_effect=_fake_retrieve
        ):
            result = download_model()
        assert result == tmp_path / ".cache" / "cortex-beam-abstain" / "model.onnx"
