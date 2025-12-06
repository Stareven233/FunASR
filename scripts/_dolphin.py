'''
cd D:\Code\projects\FunASR
$audio = "D:/Document/Audio/!raw/篠宮ゆり.aac"
$audio = "D:/Document/Audio/!raw/久岐忍/0ca749328013644d.wav -i D:/Document/Audio/!raw/rie.mp4"
$audio = "D:/Document/ai-sings/Ending Note/Ending Note 門谷純_Vocals_vocals.flac"
uv run scripts/_dolphin.py -i $audio

! 最长支持30s，超过的部分会被无视
'''

import sys
from pathlib import Path

import dolphin

sys.path.append('.')
from scripts import ROOT, parser


args = parser.parse_args()
model = dolphin.load_model('small', ROOT / 'model_zoo/dolphin-small', 'cuda')


def run(path: Path):
  # D:/Document/Audio/rie.aac
  waveform = dolphin.load_audio(path)
  print('\n\n\ndolphin最长只支持30s，需要实现分段推理！！！\n\n\n')
  # result = model(waveform)
  # # Specify language and region
  result = model(waveform, lang_sym='ja', region_sym='JP')
  print(result)  # TranscribeResult(text='<zh><CN><asr><0.00> 抱歉奎小姐我们是想来为荒龙派宣传单的时<3.96>', text_nospecial='抱歉奎小姐我们是想来为荒龙派宣传单的时', language='zh', region='CN', rtf=0.35)
  print(result.text)  # <zh><CN><asr><0.00> 抱歉奎小姐我们是想来为荒龙派宣传单的时<3.96>
  with path.with_suffix('.txt').open('w', encoding='utf-8') as f:
    f.write(result.text)

# -i D:/Document/Audio/!raw/久岐忍/0ae36ff8fc8c56e6.wav -i D:/Document/Audio/!raw/久岐忍/0ca749328013644d.wav
if __name__ == '__main__':
  for i in args.input:
    run(i)
