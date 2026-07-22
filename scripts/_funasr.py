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
uv run scripts/_funasr.py -m FunAudioLLM/Fun-ASR-Nano-2512 -i $audio -s "raw|srt"
# 混合语种 / 日文视频：保持默认 language=auto；短音频可 --no-vad
# uv run scripts/_funasr.py -m funasrNano2512 -i $audio -s "srt|ass" --title "篠宮ゆり"
# uv run scripts/_funasr.py -m funasrNano2512 -i "D:/t/h/jrcr94_oc3G6e2Zh_2026-03-22_11-34-07.mp4" -s "srt" --no-vad
'''

import json
import os
import sys
from pathlib import Path
from typing import Iterator

from funasr import AutoModel

sys.path.append('.')
from scripts import ROOT, parse_transcribe_args
from scripts.subtitles import run as build_subtitle

# ── 项目自定义配置 ──────────────────────────────────────────────────
model_dir = ROOT / 'model_zoo/models'
# vad_model_dir = model_dir / 'speech_fsmn_vad_zh-cn-16k-common-pytorch'
DEFAULT_VAD_MODEL = 'fsmn-vad'
os.environ['MODELSCOPE_CACHE'] = model_dir.parent.as_posix()

model_mapping: dict[str, str] = {
    'sensevoice': 'SenseVoiceSmall',
    'whisper': 'Whisper-large-v3-turbo',
    'funasrNano2512': 'FunAudioLLM/Fun-ASR-Nano-2512',
    'funasrNanoMlt2512': 'FunAudioLLM/Fun-ASR-MLT-Nano-2512',
}

# ── 推理 ──────────────────────────────────────────────────────────

def run(model, inputs: list[Path | str], *, language: str = 'auto', use_vad: bool = True):
    """
    调用模型生成识别结果。

    整合了官方的改进：
    - sentence_timestamp=True: 直接返回分句级别的结构化结果 (sentence_info)
    - return_time_stamps=True:  返回逐句的时间戳
    - disable_update=True:      关闭模型更新检查，减少噪音输出
    - language 默认 auto:       混合语种更稳；单语种视频可显式指定
    """
    inputs: list[str] = [i.as_posix() if isinstance(i, Path) else i for i in inputs]
    res = model.generate(
        input=inputs,
        cache={},
        # hotwords=['leaf'],
        batch_size=1,
        itn=True,
        # Fun-ASR-Nano-2512: 中文、英文、日文
        # Fun-ASR-MLT-Nano-2512: 韩文、越南语、印尼语、泰国语、马来语、菲律宾语、阿拉伯语、印地语、保加利亚语、克罗地亚语、捷克语、丹麦语、荷兰语、爱沙尼亚语、芬兰语、希腊语、匈牙利语、爱尔兰语、拉脱维亚语、立陶宛语、马耳他语、波兰语、葡萄牙语、罗马尼亚语、斯洛伐克语、斯洛文尼亚语、瑞典语
        language=language,  # 默认 auto；单语种可 --language 中文/ja/en
        use_itn=True,
        # 官方推荐参数
        sentence_timestamp=True,
        output_timestamp=True,
        return_time_stamps=True,
        batch_size_s=60,
        merge_vad=bool(use_vad),
        merge_length_s=15,
        disable_update=True,
        no_speech_threshold=0.6,
    )
    return res


# ── 输出处理 ──────────────────────────────────────────────────────

def make_output(
    output: dict,
    in_path: Path,
    subtitle_type: str | None,
    save: bool = True,
    title: str = '',
):
    """
    保存 raw JSON / 字幕。
    直接把原生 dict 交给 build_subtitle，避免 str(output) → literal_eval 的双重序列化。
    sentence_info 缺失时的回退逻辑在 scripts/subtitles.py 内处理。
    """
    def _save(out, suffix: str | None):
        if not save:
            return
        if suffix is not None:
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

    # 默认标题：输入文件名（无扩展名），可用 --title 覆盖
    ass_title = title or in_path.stem
    for s in stypes:
        ret = build_subtitle(output, 'funasr-nano', s, True, title=ass_title)
        result.append(ret)
        _save(ret, s)

    return result


# ── 入口 ──────────────────────────────────────────────────────────

def main(args):
    '''model_dir:      模型名称，或本地磁盘中的模型路径。
    vad_model:       表示开启VAD，VAD的作用是将长音频切割成短音频，此时推理耗时包括了VAD与SenseVoice总耗时，为链路耗时，如果需要单独测试SenseVoice模型耗时，可以关闭VAD模型。
    vad_kwargs:      表示VAD模型配置
    max_single_segment_time: 表示vad_model最大切割音频时长, 单位是毫秒ms。
    use_itn:         输出结果中是否包含标点与逆文本正则化。
    batch_size_s:    表示采用动态batch，batch中总音频时长，单位为秒s。
    merge_vad:       是否将 vad 模型切割的短音频碎片合成，合并后长度为merge_length_s，单位为秒s。
    ban_emo_unk:     禁用emo_unk标签，禁用后所有的句子都会被赋与情感标签。
    '''
    stype = args.subtitle_type
    model = args.model_name
    if not model:
        model = 'funasrNano2512'

    use_vad = bool(getattr(args, 'vad', True))
    language = getattr(args, 'language', None) or 'auto'
    title = getattr(args, 'title', '') or ''
    vad_model = DEFAULT_VAD_MODEL if use_vad else None

    if model in model_mapping:
        model_name = model_dir / model_mapping[model]
        hub = 'hf' if 'Fun-ASR-Nano' in model_mapping[model] else 'ms'
    else:
        model_name = model
        hub = 'hf' if 'Fun-ASR-Nano' in model else 'ms'

    model = AutoModel(
        model=model_name,
        vad_model=vad_model,
        # 最大单段时长 30s
        vad_kwargs={'max_single_segment_time': 30000} if vad_model else None,
        device='cuda:0',
        trust_remote_code=True,
        hub=hub,
        # FunASR 内部自动通过 ffmpeg 解码任意格式 (mp4/aac/flac/mp3 等) 并重采样至 16kHz，
        # 无需事先用 ffmpeg 抽音频转采样率。
    )
    print(f'[Config] language={language} vad={"on" if use_vad else "off"}')
    model_output = run(model, args.input, language=language, use_vad=use_vad)
    results: list[str | Iterator[str]] = []
    for i, o in zip(args.input, model_output):
        # 保存原始/字幕输出；原生 dict 直传，避免 str 包装
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
