"""Unit tests for cache-first / offline flags on the hub download helpers."""

from pathlib import Path

import pytest

from funasr.download import download_model_from_hub as hub


def test_get_or_download_model_dir_local_path_skips_latest_by_default(tmp_path, monkeypatch):
    model_dir = tmp_path / "local_model"
    model_dir.mkdir()
    called = {"latest": 0, "snap": 0}

    def fake_latest(*_a, **_k):
        called["latest"] += 1

    def fake_snap(*_a, **_k):
        called["snap"] += 1
        return str(model_dir)

    import modelscope.hub.check_model as cm
    import modelscope.hub.snapshot_download as sd

    monkeypatch.setattr(cm, "check_local_model_is_latest", fake_latest)
    monkeypatch.setattr(sd, "snapshot_download", fake_snap)

    out = hub.get_or_download_model_dir(str(model_dir), check_latest=False)
    assert Path(out) == model_dir
    assert called["latest"] == 0
    assert called["snap"] == 0


def test_get_or_download_model_dir_local_path_checks_latest_when_requested(tmp_path, monkeypatch):
    model_dir = tmp_path / "local_model"
    model_dir.mkdir()
    called = {"latest": 0}

    def fake_latest(*_a, **_k):
        called["latest"] += 1

    import modelscope.hub.check_model as cm
    import modelscope.hub.snapshot_download as sd

    monkeypatch.setattr(cm, "check_local_model_is_latest", fake_latest)
    monkeypatch.setattr(sd, "snapshot_download", lambda *a, **k: str(model_dir))

    out = hub.get_or_download_model_dir(
        str(model_dir), check_latest=True, local_files_only=False
    )
    assert Path(out) == model_dir
    assert called["latest"] == 1


def test_get_or_download_model_dir_local_path_skips_latest_when_local_only(tmp_path, monkeypatch):
    model_dir = tmp_path / "local_model"
    model_dir.mkdir()
    called = {"latest": 0}

    def fake_latest(*_a, **_k):
        called["latest"] += 1

    import modelscope.hub.check_model as cm

    monkeypatch.setattr(cm, "check_local_model_is_latest", fake_latest)

    out = hub.get_or_download_model_dir(
        str(model_dir), check_latest=True, local_files_only=True
    )
    assert Path(out) == model_dir
    assert called["latest"] == 0


def test_get_or_download_model_dir_forwards_flags(monkeypatch):
    seen = {}

    def fake_snap(model, **kwargs):
        seen["model"] = model
        seen["kwargs"] = kwargs
        return "/tmp/fake-ms-cache"

    import modelscope.hub.snapshot_download as sd

    monkeypatch.setattr(sd, "snapshot_download", fake_snap)

    out = hub.get_or_download_model_dir(
        "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        model_revision="v2.0.4",
        cache_dir="/tmp/ms-cache",
        local_files_only=True,
        check_latest=False,
    )
    assert out == "/tmp/fake-ms-cache"
    assert seen["model"] == "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
    assert seen["kwargs"]["revision"] == "v2.0.4"
    assert seen["kwargs"]["cache_dir"] == "/tmp/ms-cache"
    assert seen["kwargs"]["local_files_only"] is True


def test_get_or_download_model_dir_hf_forwards_flags(monkeypatch):
    seen = {}

    def fake_snap(repo_id, **kwargs):
        seen["repo_id"] = repo_id
        seen["kwargs"] = kwargs
        return "/tmp/fake-hf-cache"

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snap)

    out = hub.get_or_download_model_dir_hf(
        "FunAudioLLM/Fun-ASR-Nano-2512",
        model_revision="abc123",
        cache_dir="/tmp/hf-cache",
        local_files_only=True,
        check_latest=False,
    )
    assert out == "/tmp/fake-hf-cache"
    assert seen["repo_id"] == "FunAudioLLM/Fun-ASR-Nano-2512"
    assert seen["kwargs"]["revision"] == "abc123"
    assert seen["kwargs"]["cache_dir"] == "/tmp/hf-cache"
    assert seen["kwargs"]["local_files_only"] is True


def test_get_or_download_model_dir_hf_maps_master_to_none(monkeypatch):
    seen = {}

    def fake_snap(repo_id, **kwargs):
        seen["kwargs"] = kwargs
        return "/tmp/fake-hf-cache"

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snap)

    hub.get_or_download_model_dir_hf(
        "FunAudioLLM/Fun-ASR-Nano-2512",
        model_revision="master",
        local_files_only=False,
    )
    assert "revision" not in seen["kwargs"]


def test_get_or_download_model_dir_hf_local_path(tmp_path):
    model_dir = tmp_path / "hf_local"
    model_dir.mkdir()
    out = hub.get_or_download_model_dir_hf(str(model_dir), local_files_only=True)
    assert Path(out) == model_dir


def test_download_from_hf_raises_on_cache_miss(monkeypatch):
    def boom(*_a, **_k):
        raise FileNotFoundError("not in cache")

    monkeypatch.setattr(hub, "get_or_download_model_dir_hf", boom)

    with pytest.raises(RuntimeError) as ei:
        hub.download_from_hf(
            model="FunAudioLLM/Fun-ASR-Nano-2512",
            local_files_only=True,
            cache_dir="/tmp/empty",
            check_latest=False,
        )
    msg = str(ei.value)
    assert "HuggingFace" in msg
    assert "local_files_only=True" in msg
    assert "--allow-download" in msg


def test_download_from_ms_raises_on_cache_miss(monkeypatch):
    def boom(*_a, **_k):
        raise FileNotFoundError("not in cache")

    monkeypatch.setattr(hub, "get_or_download_model_dir", boom)

    with pytest.raises(RuntimeError) as ei:
        hub.download_from_ms(
            model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            local_files_only=True,
            cache_dir="/tmp/empty",
            check_latest=False,
        )
    msg = str(ei.value)
    assert "ModelScope" in msg
    assert "local_files_only=True" in msg


def test_download_from_hf_does_not_retry_online_after_miss(monkeypatch):
    """Cache-only miss must raise; never fall through with the bare repo id."""
    calls = {"n": 0}

    def boom(*_a, **_k):
        calls["n"] += 1
        raise OSError("offline")

    monkeypatch.setattr(hub, "get_or_download_model_dir_hf", boom)

    with pytest.raises(RuntimeError):
        hub.download_from_hf(
            model="FunAudioLLM/Fun-ASR-Nano-2512",
            local_files_only=True,
            check_latest=False,
        )
    assert calls["n"] == 1
