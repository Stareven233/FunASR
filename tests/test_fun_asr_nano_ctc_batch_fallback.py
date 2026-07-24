from types import SimpleNamespace

import torch

from funasr.models.fun_asr_nano import model as nano_model


class _FakeLLM:
    def __init__(self):
        self.config = SimpleNamespace(pad_token_id=None, eos_token_id=0)
        self.calls = 0
        self.last_batch_size = None

    def to(self, _dtype):
        return self

    def generate(self, **kwargs):
        self.calls += 1
        embeds = kwargs.get("inputs_embeds")
        if embeds is not None:
            self.last_batch_size = int(embeds.shape[0])
            # one generated token per batch row so batch_decode gets B texts
            return torch.arange(10, 10 + embeds.shape[0], dtype=torch.long).unsqueeze(-1)
        self.last_batch_size = 1
        return torch.tensor([[10 + self.calls]], dtype=torch.long)


class _FakeTokenizer:
    def batch_decode(self, generated_ids, **_kwargs):
        # generated_ids: [B, T]
        if generated_ids.ndim == 1:
            generated_ids = generated_ids.unsqueeze(0)
        return [f"text-{int(row[-1])}" for row in generated_ids]


class _FakeCTCDecoder:
    def __call__(self, encoder_out, encoder_out_lens):
        b = encoder_out.shape[0]
        t = encoder_out.shape[1]
        # shape [B, T, 3] with blank=0, token1=1, token2=2 preferred
        logits = torch.zeros(b, t, 3, dtype=torch.float32)
        logits[:, 0, 1] = 2.0
        if t > 1:
            logits[:, 1, 2] = 3.0
        return logits, encoder_out_lens


class _FakeCTC:
    def log_softmax(self, decoder_out):
        return decoder_out


class _FakeCTCTokenizer:
    def decode(self, token_ids):
        return "".join(str(token_id) for token_id in token_ids)

    def encode(self, text):
        return [1] if text else []


def _make_instance():
    instance = object.__new__(nano_model.FunASRNano)
    instance.ctc_decoder = _FakeCTCDecoder()
    instance.ctc = _FakeCTC()
    instance.ctc_tokenizer = _FakeCTCTokenizer()
    instance.blank_id = 0
    instance.llm = _FakeLLM()
    return instance


def _patch_prepare(instance, monkeypatch, *, allow_batch=False):
    prepare_calls = []

    def fake_inference_prepare(
        data_in,
        data_lengths=None,
        key=None,
        tokenizer=None,
        frontend=None,
        **_kwargs,
    ):
        if (not allow_batch) and len(data_in) > 1:
            raise NotImplementedError("batch decoding is not implemented")
        prepare_calls.append((data_in[0], key[0] if key else None))
        return (
            torch.zeros(1, 2, 4),
            {"assistant": [f"label-{data_in[0]}"]},
            {"attention_mask": torch.ones(1, 2, dtype=torch.long)},
            torch.empty(1, 0, dtype=torch.long),
            {
                "encoder_out": torch.zeros(1, 2, 3),
                "encoder_out_lens": torch.tensor([2]),
                "batch_data_time": 1.0,
            },
        )

    monkeypatch.setattr(instance, "inference_prepare", fake_inference_prepare)
    return prepare_calls


def _patch_forced_align(monkeypatch):
    monkeypatch.setattr(
        nano_model,
        "forced_align",
        lambda _logits, target_ids, _blank_id: [
            {"token": int(target_ids[0]) if len(target_ids) else 0, "start_time": 0.0, "end_time": 1.0}
        ],
    )


def test_ctc_decoder_default_timestamps_mode_batches_llm(monkeypatch):
    """Default nano_batch_mode=timestamps: one llm.generate for the pack + CTC fields."""
    instance = _make_instance()
    prepare_calls = _patch_prepare(instance, monkeypatch)
    _patch_forced_align(monkeypatch)

    results, meta = instance.inference_llm(
        ["seg-a", "seg-b"],
        key=["key-a", "key-b"],
        tokenizer=_FakeTokenizer(),
        frontend=None,
        device="cpu",
        llm_dtype="fp32",
    )

    assert prepare_calls == [("seg-a", "key-a"), ("seg-b", "key-b")]
    assert instance.llm.calls == 1
    assert instance.llm.last_batch_size == 2
    assert [result["key"] for result in results] == ["key-a", "key-b"]
    assert all("timestamps" in result for result in results)
    assert all("ctc_timestamps" in result for result in results)
    assert all("ctc_text" in result for result in results)
    assert meta["batch_data_time"] == 2.0


def test_ctc_decoder_sequential_mode_one_generate_per_segment(monkeypatch):
    """sequential mode: one llm.generate per VAD segment (oracle / debug)."""
    instance = _make_instance()
    prepare_calls = _patch_prepare(instance, monkeypatch)
    _patch_forced_align(monkeypatch)

    results, meta = instance.inference_llm(
        ["seg-a", "seg-b"],
        key=["key-a", "key-b"],
        tokenizer=_FakeTokenizer(),
        frontend=None,
        device="cpu",
        llm_dtype="fp32",
        nano_batch_mode="sequential",
    )

    assert prepare_calls == [("seg-a", "key-a"), ("seg-b", "key-b")]
    assert instance.llm.calls == 2
    assert [result["key"] for result in results] == ["key-a", "key-b"]
    assert all("timestamps" in result for result in results)
    assert all("ctc_timestamps" in result for result in results)
    assert meta["batch_data_time"] == 2.0


def test_ctc_decoder_fast_mode_no_timestamps(monkeypatch):
    """fast mode: batched LLM path, no CTC timestamp fields claimed."""
    instance = _make_instance()
    prepare_calls = _patch_prepare(instance, monkeypatch)
    _patch_forced_align(monkeypatch)

    results, meta = instance.inference_llm(
        ["seg-a", "seg-b"],
        key=["key-a", "key-b"],
        tokenizer=_FakeTokenizer(),
        frontend=None,
        device="cpu",
        llm_dtype="fp32",
        nano_batch_mode="fast",
    )

    assert prepare_calls == [("seg-a", "key-a"), ("seg-b", "key-b")]
    assert instance.llm.calls == 1
    assert instance.llm.last_batch_size == 2
    assert [result["key"] for result in results] == ["key-a", "key-b"]
    assert [result["text"] for result in results] == ["text-10", "text-11"]
    assert all("timestamps" not in result for result in results)
    assert all("ctc_timestamps" not in result for result in results)
    assert meta == {}


def test_no_ctc_decoder_always_batches(monkeypatch):
    """Without a CTC decoder, multi-segment always takes the fast batch path."""
    instance = _make_instance()
    instance.ctc_decoder = None
    prepare_calls = _patch_prepare(instance, monkeypatch)
    _patch_forced_align(monkeypatch)

    results, meta = instance.inference_llm(
        ["seg-a", "seg-b"],
        key=["key-a", "key-b"],
        tokenizer=_FakeTokenizer(),
        frontend=None,
        device="cpu",
        llm_dtype="fp32",
        nano_batch_mode="timestamps",  # ignored when no CTC
    )

    assert prepare_calls == [("seg-a", "key-a"), ("seg-b", "key-b")]
    assert instance.llm.calls == 1
    assert instance.llm.last_batch_size == 2
    assert all("timestamps" not in result for result in results)
    assert meta == {}


def test_timestamps_mode_matches_sequential_field_layout(monkeypatch):
    """timestamps and sequential should attach the same CTC field names / shapes."""
    instance_ts = _make_instance()
    instance_seq = _make_instance()
    _patch_prepare(instance_ts, monkeypatch)
    _patch_prepare(instance_seq, monkeypatch)
    _patch_forced_align(monkeypatch)

    common = dict(
        key=["key-a", "key-b"],
        tokenizer=_FakeTokenizer(),
        frontend=None,
        device="cpu",
        llm_dtype="fp32",
    )
    results_ts, _ = instance_ts.inference_llm(
        ["seg-a", "seg-b"], nano_batch_mode="timestamps", **common
    )
    results_seq, _ = instance_seq.inference_llm(
        ["seg-a", "seg-b"], nano_batch_mode="sequential", **common
    )

    for a, b in zip(results_ts, results_seq):
        assert set(a.keys()) == set(b.keys())
        assert "timestamps" in a and "ctc_timestamps" in a
        assert isinstance(a["timestamps"], list) and len(a["timestamps"]) >= 1
        assert a["timestamps"][0]["token"] == b["timestamps"][0]["token"]
        # scaled times: start_time * 6 * 10 / 1000 = 0.0, end = 0.06
        assert a["timestamps"][0]["end_time"] == b["timestamps"][0]["end_time"]


# Keep the historical name as an alias so older runners still find a test.
def test_ctc_decoder_multi_segment_input_uses_single_segment_fallback(monkeypatch):
    """Backward-compatible: multi-segment with CTC still returns ordered results.

    Historically this forced sequential fallback; the new default is timestamps
    batching, which still prepares one segment at a time and returns CTC fields.
    """
    test_ctc_decoder_default_timestamps_mode_batches_llm(monkeypatch)
