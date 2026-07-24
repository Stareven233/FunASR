r'''https://github.com/modelscope/FunASR/blob/main/README_zh.md
本仓库就是 pip3 install -U funasr 安装的pypi包

model hub: https://www.modelscope.cn/models?page=1&tasks=auto-speech-recognition <— dowload with git lfs

uv sync
uv pip install torch==2.11.0+cu130 torchaudio==2.11.0+cu130 --extra-index-url https://download.pytorch.org/whl/cu117
uv pip install openai-whisper
uv pip install -e ./

cd D:\Code\projects\FunASR
$audio = "D:/Document/Audio/!raw/篠宮ゆり.aac"
# $audio = "D:/Document/ai-sings/Ending Note/Ending Note 門谷純_Vocals_vocals.flac"
# $audio = "D:\Document\Video\leafflow\vocal\output.flac"
# 💡 支持任意格式（mp4/aac/flac/mp3/m4a/wav…），FunASR 内部自动解码并重采样至 16kHz
# VAD 开启时：按切片流式写 srt/ass（中途崩溃也不丢已识别部分）+ tqdm 进度
# 默认 cache-only（不联网校验/下载）；冷缓存请加 --allow-download
uv run scripts/_funasr.py -m FunAudioLLM/Fun-ASR-Nano-2512 -i $audio -s "raw|srt"
# 混合语种 / 日文视频：保持默认 language=auto；短音频可 --no-vad
# 长音频加速：调大 --batch-size（默认 8）与 --batch-size-s（默认 60）
# Nano 多段批处理：--nano-batch-mode timestamps|fast|sequential（默认 timestamps）
# uv run scripts/_funasr.py -m funasrNano2512 -i $audio -s "srt|ass" --title "篠宮ゆり" --batch-size 16 --batch-size-s 120
# uv run scripts/_funasr.py -m funasrNano2512 -i "D:/t/h/jrcr94_oc3G6e2Zh_2026-03-22_11-34-07.mp4" -s "srt" --no-vad
# 首次拉模型：uv run scripts/_funasr.py -m funasrNano2512 -i $audio -s srt --allow-download
'''

import json
import logging
import os
import sys
import time
from collections.abc import Iterator
from pathlib import Path

# ── 缓存 / 日志：必须在 import funasr / huggingface 之前 ──────────
# 注意：不能 from scripts import ROOT —— scripts/__init__.py 会立刻 import funasr
_ROOT = Path(__file__).resolve().parent.parent
model_dir = _ROOT / 'model_zoo/models'
_cache_root = model_dir.parent  # model_zoo/
os.environ['MODELSCOPE_CACHE'] = _cache_root.as_posix()
# HuggingFace 与 ModelScope 共用 model_zoo 作为缓存根
os.environ['HF_HOME'] = _cache_root.as_posix()
os.environ['HF_HUB_CACHE'] = (_cache_root / 'hub').as_posix()
os.environ['TRANSFORMERS_CACHE'] = (_cache_root / 'transformers').as_posix()
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 压掉 FunASR / transformers 的 INFO 刷屏；我们自己打印阶段进度
logging.getLogger().setLevel(logging.WARNING)
logging.getLogger('funasr').setLevel(logging.WARNING)
logging.getLogger('modelscope').setLevel(logging.WARNING)
logging.getLogger('transformers').setLevel(logging.ERROR)

sys.path.append('.')
from tqdm import tqdm

from funasr import AutoModel
from funasr.utils.load_utils import load_audio_text_image_video
from funasr.utils.vad_utils import merge_vad, slice_padding_audio_samples
from scripts import ROOT, parse_transcribe_args
from scripts.subtitles import ASS_Resolver, SRT_Resolver
from scripts.subtitles import run as build_subtitle

# 与 scripts.ROOT 对齐（防御性：若相对路径解析不一致则用 scripts.ROOT）
model_dir = ROOT / 'model_zoo/models'

# ── 项目自定义配置 ──────────────────────────────────────────────────
DEFAULT_VAD_MODEL = 'fsmn-vad'
# DEFAULT_VAD_MODEL = model_dir / 'iic/speech_fsmn_vad_zh-cn-16k-common-pytorch'
MERGE_LENGTH_S = 15
MAX_SINGLE_SEGMENT_MS = 30000

model_mapping: dict[str, str] = {
    'sensevoice': 'SenseVoiceSmall',
    'whisper': 'Whisper-large-v3-turbo',
    'funasrNano2512': 'FunAudioLLM/Fun-ASR-Nano-2512',
    'funasrNanoMlt2512': 'FunAudioLLM/Fun-ASR-MLT-Nano-2512',
}

# generate / inference 共用参数（字幕时间戳必需；fast 模式会去掉时间戳类 flag）
_ASR_GEN_KW = {
    'itn': True,
    'use_itn': True,
    'sentence_timestamp': True,
    'output_timestamp': True,
    'return_time_stamps': True,
    'disable_update': True,
    'no_speech_threshold': 0.6,
}

# fast 模式下不向模型索取 token 级时间戳（避免误导）
_ASR_GEN_KW_FAST = {
    'itn': True,
    'use_itn': True,
    'disable_update': True,
    'no_speech_threshold': 0.6,
}

_FAST_MODE_WARNED = False


def _asr_gen_kwargs(nano_batch_mode: str = 'timestamps') -> dict:
    """Inference kwargs for the selected Nano batch mode."""
    if (nano_batch_mode or 'timestamps').lower() == 'fast':
        return dict(_ASR_GEN_KW_FAST)
    return dict(_ASR_GEN_KW)


def _apply_vad_range_fallback(result: dict, seg: list, *, nano_batch_mode: str) -> dict:
    """In fast mode, fill missing timestamps with the VAD segment range.

    Does not invent token-accurate timing; only provides a whole-segment cue
    so SRT/ASS still emit something. Call at the result-normalization boundary
    (before LiveSubtitleSink), not inside the sink.
    """
    global _FAST_MODE_WARNED
    mode = (nano_batch_mode or 'timestamps').lower()
    if mode != 'fast':
        return result
    has_ts = bool(
        result.get('timestamps')
        or result.get('ctc_timestamps')
        or result.get('timestamp')
        or result.get('sentence_info')
    )
    if has_ts:
        return result
    if not _FAST_MODE_WARNED:
        print(
            '[Warn] nano_batch_mode=fast: no CTC/token timestamps; '
            'subtitle timing falls back to VAD segment ranges (not token-accurate).'
        )
        _FAST_MODE_WARNED = True
    start_ms = int(seg[0])
    end_ms = int(seg[1])
    # result is about to be offset by start_ms, so store relative range [0, dur]
    dur_ms = max(end_ms - start_ms, 1)
    text = (result.get('text') or '').strip()
    out = dict(result)
    out['sentence_info'] = [{'start': 0, 'end': dur_ms, 'text': text}]
    out['timestamp'] = [[0, dur_ms]]
    return out


# ── 字幕流式落盘 ──────────────────────────────────────────────────

def _offset_result(result: dict, offset_ms: int) -> dict:
    """把单段 ASR 相对时间戳平移到全曲绝对时间。"""
    if not offset_ms:
        return result
    out = dict(result)
    off_s = offset_ms / 1000.0

    def _shift_pair(pair):
        if isinstance(pair, dict):
            t = dict(pair)
            for sk, ek in (
                ('start_time', 'end_time'),
                ('start', 'end'),
            ):
                if sk in t and t[sk] is not None:
                    v = float(t[sk])
                    # Nano token 时间多为秒；毫秒级数值则按 ms 加
                    t[sk] = v + (off_s if v < 1000 else offset_ms)
                if ek in t and t[ek] is not None:
                    v = float(t[ek])
                    t[ek] = v + (off_s if v < 10000 else offset_ms)
            return t
        if isinstance(pair, (list, tuple)) and len(pair) >= 2:
            a, b = float(pair[0]), float(pair[1])
            # list 格式官方按 ms
            return [int(a) + offset_ms, int(b) + offset_ms, *pair[2:]]
        return pair

    for key in ('timestamp', 'timestamps', 'ctc_timestamps'):
        if out.get(key):
            out[key] = [_shift_pair(t) for t in out[key]]

    if out.get('sentence_info'):
        si = []
        for seg in out['sentence_info']:
            s = dict(seg)
            if 'start' in s and s['start'] is not None:
                s['start'] = int(s['start']) + offset_ms
            if 'end' in s and s['end'] is not None:
                s['end'] = int(s['end']) + offset_ms
            if s.get('timestamp'):
                s['timestamp'] = [_shift_pair(t) for t in s['timestamp']]
            si.append(s)
        out['sentence_info'] = si
    return out


def _merge_segment_results(segments: list[dict], key: str = '') -> dict:
    """把流式分段结果合成一份完整 raw dict（给 json 输出）。"""
    if not segments:
        return {'key': key, 'text': '', 'timestamps': [], 'timestamp': []}
    texts = []
    timestamps = []
    ctc_timestamps = []
    timestamp = []
    sentence_info = []
    for seg in segments:
        t = (seg.get('text') or '').strip()
        if t:
            texts.append(t)
        if seg.get('timestamps'):
            timestamps.extend(seg['timestamps'])
        if seg.get('ctc_timestamps'):
            ctc_timestamps.extend(seg['ctc_timestamps'])
        if seg.get('timestamp'):
            timestamp.extend(seg['timestamp'])
        if seg.get('sentence_info'):
            sentence_info.extend(seg['sentence_info'])
    out = {
        'key': key or segments[0].get('key', ''),
        'text': ' '.join(texts),
    }
    if timestamps:
        out['timestamps'] = timestamps
    if ctc_timestamps:
        out['ctc_timestamps'] = ctc_timestamps
    if timestamp:
        out['timestamp'] = timestamp
    if sentence_info:
        out['sentence_info'] = sentence_info
    # 兼容字段
    for k in ('text_tn', 'label', 'ctc_text'):
        vals = [s[k] for s in segments if s.get(k)]
        if vals:
            out[k] = ' '.join(str(v) for v in vals)
    return out


class LiveSubtitleSink:
    """
    VAD 切片完成后立即追加写入字幕。
    - srt/ass: 打开即写 header，每段 ASR 完就 flush 新 cue
    - raw/json: 收齐后一次性写（结构完整）
    中途崩溃时 srt/ass 已落盘的部分仍可播放。
    """

    def __init__(
        self,
        in_path: Path,
        subtitle_type: str | None,
        *,
        title: str = '',
        save: bool = True,
    ):
        self.in_path = Path(in_path)
        self.save = save
        self.title = title or self.in_path.stem
        stypes = set([] if subtitle_type is None else subtitle_type.split('|'))
        self.want_raw = len(stypes) == 0 or 'raw' in stypes
        stypes.discard('raw')
        self.sub_types = {s for s in stypes if s in ('srt', 'ass')}
        # 未知类型仍走一次性 build
        self.extra_types = stypes - self.sub_types

        self._files: dict[str, object] = {}
        self._index: dict[str, int] = {}  # 各字幕类型独立序号
        self._resolvers: dict[str, SRT_Resolver] = {}
        self.segments: list[dict] = []
        self.cue_count = 0

        if self.save:
            for s in self.sub_types:
                cls = ASS_Resolver if s == 'ass' else SRT_Resolver
                # 空 content 只拿 format_pre / format_line
                r = cls({}, title=self.title, input_type='funasr-nano')
                r.timelines = []
                self._resolvers[s] = r
                self._index[s] = 0
                p = self.in_path.with_suffix(f'.{s}')
                f = p.open('w', encoding='utf-8')
                pre = r.format_pre()
                if pre:
                    f.write(pre)
                    f.flush()
                self._files[s] = f
                print(f'[Stream] open {p.as_posix()}')

    def append_segment(self, result: dict):
        """追加一段已做绝对时间偏移的 ASR 结果。"""
        self.segments.append(result)
        if not self.save or not self.sub_types:
            return
        # 只对第一种字幕类型计 cue，避免 srt|ass 双开时重复计数
        counted = False
        for s, resolver in self._resolvers.items():
            lines = resolver.resolve_funasr_nano(result)
            f = self._files[s]
            n_written = 0
            for st, et, text in lines:
                if not text:
                    continue
                self._index[s] += 1
                st_t = resolver.format_seconds(st)
                et_t = resolver.format_seconds(et)
                f.write(resolver.format_line(self._index[s], st_t, et_t, text))
                n_written += 1
            if not counted and n_written:
                self.cue_count += n_written
                counted = True
            f.flush()

    def finalize(self) -> dict:
        """收尾：关字幕文件、写 raw json、处理未知类型。"""
        for f in self._files.values():
            try:
                f.close()
            except Exception:
                pass
        self._files.clear()

        merged = _merge_segment_results(self.segments, key=self.in_path.stem)
        if self.save and self.want_raw:
            p = self.in_path.with_suffix('.json')
            p.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding='utf-8')
            print(f'[Save] {p.as_posix()}  segments={len(self.segments)} cues={self.cue_count}')
        for s in self.sub_types:
            if self.save:
                print(f'[Save] {self.in_path.with_suffix(f".{s}").as_posix()}  cues≈{self.cue_count}')
        if self.save:
            for s in self.extra_types:
                ret = build_subtitle(merged, 'funasr-nano', s, False, title=self.title)
                p = self.in_path.with_suffix(f'.{s}')
                p.write_text(ret if isinstance(ret, str) else ''.join(ret), encoding='utf-8')
                print(f'[Save] {p.as_posix()}')
        return merged


# ── 推理 ──────────────────────────────────────────────────────────

def run_once(
    model,
    inputs: list[Path | str],
    *,
    language: str = 'auto',
    batch_size: int = 8,
    batch_size_s: int = 60,
    nano_batch_mode: str = 'timestamps',
):
    """无 VAD：整文件一次 generate（短音频 / --no-vad）。"""
    inputs = [i.as_posix() if isinstance(i, Path) else i for i in inputs]
    print(
        f'[ASR] start  files={len(inputs)} (no-vad) batch_size={batch_size} '
        f'nano_batch_mode={nano_batch_mode}'
    )
    t0 = time.perf_counter()
    res = model.generate(
        input=inputs,
        cache={},
        batch_size=batch_size,
        batch_size_s=batch_size_s,
        language=language,
        merge_vad=False,
        nano_batch_mode=nano_batch_mode,
        **_asr_gen_kwargs(nano_batch_mode),
    )
    print(f'[ASR] done   elapsed={time.perf_counter() - t0:.1f}s')
    return res


def run_streaming_vad(
    model,
    in_path: Path,
    *,
    language: str = 'auto',
    batch_size: int = 8,
    batch_size_s: int = 60,
    subtitle_type: str | None = None,
    title: str = '',
    save: bool = True,
    nano_batch_mode: str = 'timestamps',
) -> dict:
    """
    自建 VAD → ASR 管线，按时间顺序处理切片，每完成一批就追加写字幕。

    与官方 inference_with_vad 差异：
    - 不按长度重排（保证写出顺序 = 时间顺序，便于实时预览）
    - 用 tqdm 显示切片进度
    - LiveSubtitleSink 实时 flush srt/ass
    """
    if model.vad_model is None:
        raise RuntimeError('run_streaming_vad requires vad_model')

    path_str = in_path.as_posix() if isinstance(in_path, Path) else str(in_path)
    in_path = Path(path_str)
    sink = LiveSubtitleSink(in_path, subtitle_type, title=title, save=save)

    # ── 1) VAD ──
    print(f'[VAD] {in_path.name} ...')
    t_vad = time.perf_counter()
    vad_out = model.inference(
        path_str,
        model=model.vad_model,
        kwargs=model.vad_kwargs,
    )
    vad_segments = (vad_out[0].get('value') if vad_out else None) or []
    if MERGE_LENGTH_S > 0 and vad_segments:
        vad_segments = merge_vad(vad_segments, MERGE_LENGTH_S * 1000)
    print(f'[VAD] done  segs={len(vad_segments)}  elapsed={time.perf_counter() - t_vad:.1f}s')

    if not vad_segments:
        print('[ASR] empty speech, skip')
        return sink.finalize()

    # ── 2) 加载整段音频 ──
    fs = model.kwargs['frontend'].fs if hasattr(model.kwargs.get('frontend'), 'fs') else 16000
    speech = load_audio_text_image_video(path_str, fs=fs, audio_fs=model.kwargs.get('fs', 16000))
    speech_lengths = len(speech)
    speech_s = speech_lengths / float(fs)

    # ── 3) 按时间顺序动态打包（batch_size 段数上限 + batch_size_s 时长上限）──
    pack_limit_ms = max(int(batch_size_s) * 1000, 1)
    packs: list[list[tuple[list, int]]] = []  # each: [( [start_ms,end_ms], orig_idx ), ...]
    cur: list[tuple[list, int]] = []
    cur_ms = 0
    for idx, seg in enumerate(vad_segments):
        dur = int(seg[1]) - int(seg[0])
        if cur and (len(cur) >= batch_size or cur_ms + dur > pack_limit_ms):
            packs.append(cur)
            cur, cur_ms = [], 0
        cur.append((seg, idx))
        cur_ms += dur
    if cur:
        packs.append(cur)

    print(
        f'[ASR] start  segs={len(vad_segments)} packs={len(packs)} '
        f'audio={speech_s:.1f}s batch_size={batch_size} batch_size_s={batch_size_s} '
        f'nano_batch_mode={nano_batch_mode}'
    )
    t_asr = time.perf_counter()
    asr_cfg = dict(
        language=language,
        nano_batch_mode=nano_batch_mode,
        **_asr_gen_kwargs(nano_batch_mode),
    )

    with tqdm(
        total=len(vad_segments),
        unit='seg',
        desc='ASR',
        dynamic_ncols=True,
        mininterval=0.5,
    ) as pbar:
        for pack in packs:
            speech_j, _ = slice_padding_audio_samples(speech, speech_lengths, pack)
            # 让 inference 内部一次吃完整包，避免再被 batch_size=1 拆碎
            results = model.inference(
                speech_j,
                input_len=None,
                model=model.model,
                kwargs=model.kwargs,
                batch_size=max(len(speech_j), 1),
                **asr_cfg,
            )
            if not results:
                pbar.update(len(pack))
                continue
            for (seg, _), res in zip(pack, results):
                # fast 模式：在 offset 之前用 VAD 相对区间补时间戳
                res = _apply_vad_range_fallback(res, seg, nano_batch_mode=nano_batch_mode)
                offset_ms = int(seg[0])
                res_abs = _offset_result(res, offset_ms)
                sink.append_segment(res_abs)
                snippet = (res_abs.get('text') or '').replace('\n', ' ').strip()
                if len(snippet) > 24:
                    snippet = snippet[:24] + '…'
                pbar.set_postfix_str(snippet, refresh=False)
                pbar.update(1)

    elapsed = time.perf_counter() - t_asr
    rtf = elapsed / speech_s if speech_s > 0 else 0.0
    print(f'[ASR] done   elapsed={elapsed:.1f}s  rtf={rtf:.3f}  cues={sink.cue_count}')
    return sink.finalize()


def make_output(
    output: dict,
    in_path: Path,
    subtitle_type: str | None,
    save: bool = True,
    title: str = '',
):
    """无 VAD / 一次性结果：保存 raw JSON / 字幕。"""
    def _save(out, suffix: str | None):
        if not save or suffix is None:
            return
        p = in_path.with_suffix(f'.{suffix}')
        with p.open('w', encoding='utf-8') as f:
            if isinstance(out, Iterator):
                for o in out:
                    f.write(o)
            else:
                f.write(out)
        print('[Save]', p.as_posix())

    stypes = set([] if subtitle_type is None else subtitle_type.split('|'))
    result = []
    if len(stypes) == 0 or 'raw' in stypes:
        ret = json.dumps(output, ensure_ascii=False, indent=2)
        result.append(ret)
        _save(ret, 'json')
    stypes.discard('raw')

    ass_title = title or in_path.stem
    for s in stypes:
        ret = build_subtitle(output, 'funasr-nano', s, True, title=ass_title)
        result.append(ret)
        _save(ret, s)
    return result


# ── 入口 ──────────────────────────────────────────────────────────

def main(args):
    '''model_dir:      模型名称，或本地磁盘中的模型路径。
    vad_model:       表示开启VAD，VAD的作用是将长音频切割成短音频。
    batch_size:      ASR 每次送入的段数；流式 VAD 路径下也是每包段数上限。
    batch_size_s:    动态 batch 总音频时长（秒）。
    nano_batch_mode: Fun-ASR-Nano 多段 LLM 批处理策略（timestamps/fast/sequential）。
    allow_download:  是否允许 hub 下载 / 远程校验（默认关，缓存优先）。
    '''
    stype = args.subtitle_type
    model = args.model_name
    if not model:
        model = 'funasrNano2512'

    use_vad = bool(getattr(args, 'vad', True))
    language = getattr(args, 'language', None) or 'auto'
    title = getattr(args, 'title', '') or ''
    batch_size = max(1, int(getattr(args, 'batch_size', 8) or 8))
    batch_size_s = max(1, int(getattr(args, 'batch_size_s', 60) or 60))
    nano_batch_mode = (getattr(args, 'nano_batch_mode', None) or 'timestamps').lower()
    allow_download = bool(getattr(args, 'allow_download', False))
    check_latest = bool(getattr(args, 'check_latest', False))
    model_revision = getattr(args, 'model_revision', None)
    vad_model = DEFAULT_VAD_MODEL if use_vad else None

    if model in model_mapping:
        model_name = model_dir / model_mapping[model]
        hub = 'hf' if 'Fun-ASR-Nano' in model_mapping[model] else 'ms'
    else:
        model_name = model
        hub = 'hf' if 'Fun-ASR-Nano' in model else 'ms'

    # If mapping resolved to a local path that doesn't exist, fall back to hub id
    # so cache_dir + local_files_only can still locate the snapshot under model_zoo.
    model_arg = model_name
    if isinstance(model_name, Path):
        if model_name.exists():
            model_arg = model_name.as_posix()
        else:
            # Prefer the hub id (model_mapping value) so snapshot_download can
            # resolve from HF/MS cache under model_zoo without re-downloading.
            mapped = model_mapping.get(model, model)
            model_arg = mapped if isinstance(mapped, str) else model_name.as_posix()

    cache_policy = 'online' if allow_download else 'cache-only'
    print(
        f'[Config] language={language} vad={"on" if use_vad else "off"} '
        f'stream={"on" if use_vad else "off"} '
        f'batch_size={batch_size} batch_size_s={batch_size_s}s '
        f'nano_batch_mode={nano_batch_mode} '
        f'cache={os.environ["MODELSCOPE_CACHE"]} policy={cache_policy} '
        f'check_latest={check_latest}'
    )
    print('[Model] loading...')
    t_load = time.perf_counter()
    auto_kwargs = dict(
        model=model_arg,
        vad_model=vad_model,
        vad_kwargs={'max_single_segment_time': MAX_SINGLE_SEGMENT_MS} if vad_model else None,
        device='cuda:0',
        trust_remote_code=True,
        hub=hub,
        # 关掉 FunASR 内置 tqdm；我们自己用切片级进度条
        disable_pbar=True,
        disable_update=True,
        log_level='ERROR',
        # cache-first / offline policy (hub helpers forward these)
        cache_dir=os.environ.get('MODELSCOPE_CACHE') or _cache_root.as_posix(),
        local_files_only=not allow_download,
        check_latest=check_latest,
    )
    if model_revision:
        auto_kwargs['model_revision'] = model_revision
        if vad_model:
            auto_kwargs['vad_model_revision'] = model_revision
    try:
        model = AutoModel(**auto_kwargs)
    except Exception as e:
        if not allow_download:
            raise RuntimeError(
                f'Failed to load model from local cache only.\n'
                f'  model={model_arg!r}\n'
                f'  cache={auto_kwargs["cache_dir"]!r}\n'
                f'  hub={hub}\n'
                f'Re-run with --allow-download to fetch missing artifacts, '
                f'or ensure the model is fully cached under model_zoo.\n'
                f'Original error: {e}'
            ) from e
        raise
    print(f'[Model] ready  load={time.perf_counter() - t_load:.1f}s')

    # 有 VAD：流式写字幕；无 VAD：整文件一次出结果
    if use_vad and model.vad_model is not None:
        for in_path in args.input:
            run_streaming_vad(
                model,
                in_path,
                language=language,
                batch_size=batch_size,
                batch_size_s=batch_size_s,
                subtitle_type=stype,
                title=title,
                save=args.save_to_file,
                nano_batch_mode=nano_batch_mode,
            )
        return

    model_output = run_once(
        model,
        args.input,
        language=language,
        batch_size=batch_size,
        batch_size_s=batch_size_s,
        nano_batch_mode=nano_batch_mode,
    )
    results: list[str | Iterator[str]] = []
    for i, o in zip(args.input, model_output):
        r = make_output(o, i, stype, args.save_to_file, title=title)
        results.extend(r)

    if args.save_to_file:
        return

    for i, content in enumerate(results):
        if i > 0:
            print()
        print(f'[{i}]')
        final_chunk = content
        if isinstance(content, Iterator):
            for c in content:
                print(c, end='')
            final_chunk = c
        else:
            print(content, end='')
        if not final_chunk.endswith('\n'):
            print()


if __name__ == '__main__':
    args = parse_transcribe_args()
    main(args)


# ── 以下为官方示例残留代码 ─────────────────────────────────────────
# # ------paraformer-zh-------
# # paraformer-zh is a multi-functional asr model
# # use vad, punc, spk or not as you need
# model = AutoModel(
#     model='paraformer-zh',
#     vad_model='fsmn-vad',
#     punc_model='ct-punc',
#     # spk_model='cam++'
# )
# res = model.generate(input=f'{model.model_path}/example/asr_example.wav', batch_size_s=300, hotword='魔搭')
# print(res)
# # ------paraformer-zh-------
