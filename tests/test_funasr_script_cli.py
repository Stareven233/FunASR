"""Mocked unit tests for scripts/_funasr.py CLI defaults and streaming packing."""

import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _import_funasr_script(monkeypatch):
    """Import scripts._funasr with heavy deps stubbed so tests stay offline/CPU-only."""
    # Ensure scripts package path is importable from repo root
    repo = Path(__file__).resolve().parents[1]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    # Stub funasr modules that the script imports at module level if not already loaded
    # We rely on the real package being installed (editable); only AutoModel is mocked
    # at call sites below.
    if "scripts._funasr" in sys.modules:
        del sys.modules["scripts._funasr"]
    mod = importlib.import_module("scripts._funasr")
    return mod


def test_asr_gen_kwargs_fast_omits_timestamp_flags():
    mod = _import_funasr_script(None)
    ts = mod._asr_gen_kwargs("timestamps")
    fast = mod._asr_gen_kwargs("fast")
    assert ts.get("output_timestamp") is True
    assert ts.get("return_time_stamps") is True
    assert "output_timestamp" not in fast
    assert "return_time_stamps" not in fast
    assert "sentence_timestamp" not in fast


def test_apply_vad_range_fallback_only_in_fast_mode():
    mod = _import_funasr_script(None)
    mod._FAST_MODE_WARNED = False
    result = {"text": "hello", "key": "k"}
    seg = [1000, 2500]

    # timestamps mode: unchanged
    out = mod._apply_vad_range_fallback(dict(result), seg, nano_batch_mode="timestamps")
    assert "sentence_info" not in out

    # fast mode: inject VAD-relative range
    out = mod._apply_vad_range_fallback(dict(result), seg, nano_batch_mode="fast")
    assert out["sentence_info"][0]["start"] == 0
    assert out["sentence_info"][0]["end"] == 1500
    assert out["sentence_info"][0]["text"] == "hello"
    assert out["timestamp"] == [[0, 1500]]

    # already has timestamps: leave alone
    with_ts = {"text": "x", "timestamps": [{"token": "x", "start_time": 0.1, "end_time": 0.2}]}
    out2 = mod._apply_vad_range_fallback(with_ts, seg, nano_batch_mode="fast")
    assert "sentence_info" not in out2


def test_parse_transcribe_args_defaults(monkeypatch):
    """CLI defaults: cache-only, no check-latest, nano_batch_mode=timestamps."""
    from scripts import parse_transcribe_args
    import argparse

    # Re-build parser by calling the real function with a controlled argv
    monkeypatch.setattr(
        sys,
        "argv",
        ["_funasr.py", "-i", "dummy.wav", "-m", "funasrNano2512"],
    )
    args = parse_transcribe_args()
    assert args.allow_download is False
    assert args.check_latest is False
    assert args.nano_batch_mode == "timestamps"
    assert args.batch_size == 8
    assert args.batch_size_s == 60


def test_parse_transcribe_args_opt_in_flags(monkeypatch):
    from scripts import parse_transcribe_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "_funasr.py",
            "-i",
            "dummy.wav",
            "--allow-download",
            "--check-latest",
            "--nano-batch-mode",
            "fast",
            "--batch-size",
            "16",
            "--batch-size-s",
            "120",
        ],
    )
    args = parse_transcribe_args()
    assert args.allow_download is True
    assert args.check_latest is True
    assert args.nano_batch_mode == "fast"
    assert args.batch_size == 16
    assert args.batch_size_s == 120


def test_main_forwards_cache_policy_and_nano_mode(monkeypatch, tmp_path, capsys):
    mod = _import_funasr_script(monkeypatch)
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")

    captured = {}

    class FakeAutoModel:
        def __init__(self, **kwargs):
            captured["auto"] = kwargs
            self.vad_model = None  # force no-vad path
            self.vad_kwargs = {}
            self.kwargs = {}
            self.model = object()

        def generate(self, **kwargs):
            captured["generate"] = kwargs
            return [{"key": "a", "text": "hi"}]

    args = SimpleNamespace(
        input=[audio],
        model_name="funasrNano2512",
        subtitle_type=None,
        vad=False,
        language="auto",
        title="",
        batch_size=16,
        batch_size_s=120,
        save_to_file=False,
        allow_download=False,
        check_latest=False,
        model_revision=None,
        nano_batch_mode="timestamps",
    )

    monkeypatch.setattr(mod, "AutoModel", FakeAutoModel)
    monkeypatch.setattr(mod, "make_output", lambda *a, **k: ["ok"])

    mod.main(args)

    auto = captured["auto"]
    assert auto["local_files_only"] is True
    assert auto["check_latest"] is False
    assert auto["cache_dir"]
    gen = captured["generate"]
    assert gen["nano_batch_mode"] == "timestamps"
    assert gen.get("output_timestamp") is True

    out = capsys.readouterr().out
    assert "nano_batch_mode=timestamps" in out
    assert "policy=cache-only" in out


def test_main_allow_download_sets_online(monkeypatch, tmp_path):
    mod = _import_funasr_script(monkeypatch)
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    captured = {}

    class FakeAutoModel:
        def __init__(self, **kwargs):
            captured["auto"] = kwargs
            self.vad_model = None
            self.vad_kwargs = {}
            self.kwargs = {}
            self.model = object()

        def generate(self, **kwargs):
            captured["generate"] = kwargs
            return [{"key": "a", "text": "hi"}]

    args = SimpleNamespace(
        input=[audio],
        model_name="FunAudioLLM/Fun-ASR-Nano-2512",
        subtitle_type=None,
        vad=False,
        language="auto",
        title="",
        batch_size=8,
        batch_size_s=60,
        save_to_file=False,
        allow_download=True,
        check_latest=True,
        model_revision="abc",
        nano_batch_mode="fast",
    )
    monkeypatch.setattr(mod, "AutoModel", FakeAutoModel)
    monkeypatch.setattr(mod, "make_output", lambda *a, **k: ["ok"])
    mod.main(args)

    assert captured["auto"]["local_files_only"] is False
    assert captured["auto"]["check_latest"] is True
    assert captured["auto"]["model_revision"] == "abc"
    assert captured["generate"]["nano_batch_mode"] == "fast"
    assert "output_timestamp" not in captured["generate"]


def test_main_cache_miss_mentions_allow_download(monkeypatch, tmp_path):
    mod = _import_funasr_script(monkeypatch)
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")

    class BoomAutoModel:
        def __init__(self, **kwargs):
            raise FileNotFoundError("not cached")

    args = SimpleNamespace(
        input=[audio],
        model_name="funasrNano2512",
        subtitle_type=None,
        vad=False,
        language="auto",
        title="",
        batch_size=8,
        batch_size_s=60,
        save_to_file=False,
        allow_download=False,
        check_latest=False,
        model_revision=None,
        nano_batch_mode="timestamps",
    )
    monkeypatch.setattr(mod, "AutoModel", BoomAutoModel)
    with pytest.raises(RuntimeError) as ei:
        mod.main(args)
    assert "--allow-download" in str(ei.value)


def test_run_streaming_vad_packs_and_forwards_mode(monkeypatch, tmp_path):
    mod = _import_funasr_script(monkeypatch)
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")

    # 5 segments, batch_size=2 → at least 3 packs
    vad_segments = [[0, 1000], [1000, 2000], [2000, 3000], [3000, 4000], [4000, 5000]]
    inference_calls = []

    class FakeModel:
        def __init__(self):
            self.vad_model = object()
            self.vad_kwargs = {}
            self.model = object()
            self.kwargs = {"frontend": SimpleNamespace(fs=16000), "fs": 16000}

        def inference(self, data, **kwargs):
            # VAD path: model=vad_model
            if kwargs.get("model") is self.vad_model:
                return [{"value": vad_segments}]
            inference_calls.append({"data_len": len(data) if isinstance(data, list) else 1, **{
                k: kwargs[k] for k in ("batch_size", "nano_batch_mode", "output_timestamp")
                if k in kwargs
            }})
            n = len(data) if isinstance(data, list) else 1
            return [
                {
                    "key": f"k{i}",
                    "text": f"t{i}",
                    "timestamps": [{"token": "t", "start_time": 0.0, "end_time": 0.1}],
                }
                for i in range(n)
            ]

    monkeypatch.setattr(
        mod,
        "load_audio_text_image_video",
        lambda *_a, **_k: [0.0] * 16000 * 6,
    )
    monkeypatch.setattr(
        mod,
        "slice_padding_audio_samples",
        lambda speech, speech_lengths, pack: ([f"seg{i}" for i in range(len(pack))], None),
    )
    monkeypatch.setattr(mod, "merge_vad", lambda segs, *_a, **_k: segs)

    result = mod.run_streaming_vad(
        FakeModel(),
        audio,
        batch_size=2,
        batch_size_s=60,
        subtitle_type=None,
        save=False,
        nano_batch_mode="timestamps",
    )
    assert len(inference_calls) == 3  # 2+2+1
    assert all(c["nano_batch_mode"] == "timestamps" for c in inference_calls)
    assert inference_calls[0]["batch_size"] == 2
    assert inference_calls[0].get("output_timestamp") is True
    assert result["text"]  # merged non-empty
