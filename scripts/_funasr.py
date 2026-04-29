r'''https://github.com/modelscope/FunASR/blob/main/README_zh.md
本仓库就是 pip3 install -U funasr 安装的pypi包

model hub: https://www.modelscope.cn/models?page=1&tasks=auto-speech-recognition  <— dowload with git lfs

uv sync
uv pip install torch==2.11.0+cu130 torchaudio==2.11.0+cu130 --extra-index-url https://download.pytorch.org/whl/cu117
uv pip install openai-whisper
uv pip install -e ./

cd D:\Code\projects\FunASR
$audio = "D:/Document/Audio/!raw/篠宮ゆり.aac"
$audio = "D:/Document/ai-sings/Ending Note/Ending Note 門谷純_Vocals_vocals.flac"
$audio = "D:\Document\Video\leafflow\vocal\output.flac"
uv run scripts/_funasr.py -m FunAudioLLM/Fun-ASR-Nano-2512 -i $audio -s "raw|srt"
'''

import sys
import os
import json
from pathlib import Path
from typing import Iterator

from funasr import AutoModel

sys.path.append('.')
from scripts import ROOT, parse_transcribe_args
from scripts.subtitles import run as build_subtitle


model_dir = ROOT / 'model_zoo/models'
vad_model_dir = model_dir / 'speech_fsmn_vad_zh-cn-16k-common-pytorch'
os.environ['MODELSCOPE_CACHE'] = model_dir.parent.as_posix()


model_mapping = {
  'sensevoice': 'SenseVoiceSmall',
  # 对whisper支持不全，一坨！
  'whisper': 'Whisper-large-v3-turbo',
  'funasrNano2512': 'FunAudioLLM/Fun-ASR-Nano-2512',
}

DecodingOptions = {
  'task': 'transcribe',
  'language': None,
  'beam_size': None,
  'fp16': True,
  'without_timestamps': False,
  'prompt': None,
}


def run(model, mname, inputs:list[Path|str], vad_model=None):
  inputs: list[str] = [i.as_posix() if isinstance(i, Path) else i for i in inputs]
  is_whisper = mname == 'whisper'
  res = model.generate(
    input=inputs,
    # DecodingOptions=DecodingOptions,
    # input=f'D:/Document/Audio/rie.aac',
    # input=f'https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav',
    cache={},
    hotwords=['leaf'],
    batch_size=1,
    itn=True,
    # Fun-ASR-Nano-2512: 中文、英文、日文
    # Fun-ASR-MLT-Nano-2512: 韩文、越南语、印尼语、泰语、马来语、菲律宾语、阿拉伯语、印地语、保加利亚语、克罗地亚语、捷克语、丹麦语、荷兰语、爱沙尼亚语、芬兰语、希腊语、匈牙利语、爱尔兰语、拉脱维亚语、立陶宛语、马耳他语、波兰语、葡萄牙语、罗马尼亚语、斯洛伐克语、斯洛文尼亚语、瑞典语
    language='中文',
    # language='auto',  # 'zn', 'en', 'yue', 'ja', 'ko', 'nospeech'
    use_itn=True,
    batch_size_s=0 if is_whisper else 60,
    merge_vad=vad_model and True,
    merge_length_s=15,
  )
  return res
  # print(res)
  # old: [{'key': 'en', 'text': '<|en|><|NEUTRAL|><|Speech|><|withitn|>The tribal chieftain called for the boy and presented him with 50 pieces of gold.'}]
  # funasr-nano: [{'key': 'output', 'text': '首先解压下载的压缩包。 然后', 'text_tn': '首先解压下载的压缩包 sil 然后', 'label': 'null', 'ctc_text': '首先解压下载的压缩包  然后', 'ctc_timestamps': [{'token': '首', 'start_time': 1.26, 'end_time': 1.32, 'score': 0.998}, {'token': '先', 'start_time': 1.44, 'end_time': 1.5, 'score': 1.0}, {'token': '解', 'start_time': 1.62, 'end_time': 1.68, 'score': 0.999}, {'token': '压', 'start_time': 1.8, 'end_time': 1.86, 'score': 0.991}, {'token': '下', 'start_time': 2.04, 'end_time': 2.1, 'score': 1.0}, {'token': '载', 'start_time': 2.16, 'end_time': 2.22, 'score': 1.0}, {'token': '的', 'start_time': 2.28, 'end_time': 2.34, 'score': 0.64}, {'token': '压', 'start_time': 2.46, 'end_time': 2.52, 'score': 0.993}, {'token': '缩', 'start_time': 2.64, 'end_time': 2.7, 'score': 0.986}, {'token': '包', 'start_time': 2.82, 'end_time': 2.88, 'score': 0.999}, {'token': ' ', 'start_time': 3.18, 'end_time': 3.3, 'score': 0.946}, {'token': ' ', 'start_time': 8.52, 'end_time': 8.58, 'score': 0.706}, {'token': '然', 'start_time': 9.0, 'end_time': 9.06, 'score': 0.997}, {'token': '后', 'start_time': 9.12, 'end_time': 9.18, 'score': 0.991}]}]
  # text = rich_transcription_postprocess(res[0]['text'])
  # print(text)


def make_output(output: dict, in_path: Path, subtitle_type: str | None, save: bool = True):
  def _save(out, suffix: str|None):
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

  for s in stypes:
    ret = build_subtitle(str(output), 'funasr-nano', s, True)
    result.append(ret)
    _save(ret, s)

  return result


def main(args):
  '''
  model_dir: 模型名称，或本地磁盘中的模型路径。
  vad_model: 表示开启VAD，VAD的作用是将长音频切割成短音频，此时推理耗时包括了VAD与SenseVoice总耗时，为链路耗时，如果需要单独测试SenseVoice模型耗时，可以关闭VAD模型。
  vad_kwargs: 表示VAD模型配置
  max_single_segment_time: 表示vad_model最大切割音频时长, 单位是毫秒ms。
  use_itn: 输出结果中是否包含标点与逆文本正则化。
  batch_size_s: 表示采用动态batch，batch中总音频时长，单位为秒s。
  merge_vad: 是否将 vad 模型切割的短音频碎片合成，合并后长度为merge_length_s，单位为秒s。
  ban_emo_unk: 禁用emo_unk标签，禁用后所有的句子都会被赋与情感标签。
  '''
  stype = args.subtitle_type
  model = args.model_name
  if not model:
    model = 'funasrNano2512'

  if model in model_mapping:
    model = model_dir / model_mapping[model]
  model = AutoModel(
    model=model,
    vad_model=vad_model_dir,
    # 最大单段时长 30s
    vad_kwargs=vad_model_dir and {'max_single_segment_time': 30000},
    device='cuda:0',
    trust_remote_code=True,
    remote_code='funasr/models/fun_asr_nano/model.py'
    # hub：download models from ms (for ModelScope) or hf (for Hugging Face).
  )
  mname = model.stem if isinstance(model, Path) else model
  model_output = run(model, mname, args.input, vad_model_dir)
  results: list[str | Iterator[str]] = []
  for i, o in zip(args.input, model_output):
    # 保存原始/字幕输出
    r = make_output(o, i, stype, args.save_to_file)
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
