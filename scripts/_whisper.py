'''https://github.com/SYSTRAN/faster-whisper

models: https://huggingface.co/collections/Systran/faster-whisper
cudnn: https://github.com/Purfview/whisper-standalone-win/releases/tag/libs

cd D:\Code\projects\FunASR
$audio = "D:/Document/Audio/!raw/篠宮ゆり.aac"
$audio = "D:/Document/ai-sings/Ending Note/Ending Note 門谷純_Vocals_vocals.flac"
uv run scripts/_whisper.py -i $audio
'''
# import os
# os.environ['LD_LIBRARY_PATH'] = r'/whisper/lib/python3.10/site-packages/nvidia/cublas/lib:/whisper/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH'
# os.environ['HF_ENDPOINT'] = r'https://hf-mirror.com'
import sys
from pathlib import Path
import traceback
from tqdm import tqdm

from faster_whisper import WhisperModel
# from ..faster_whisper.transcribe import WhisperModel
sys.path.append('.')
from scripts import ROOT, parser
from subtitles import ASS_Resolver


ROOT = Path(__file__).parent.parent
args, left_argv = parser.parse_known_args()
paths: tuple[Path] = args.input

resolver = ASS_Resolver('')

# Run on GPU with FP16
# model_path = 'JhonVanced/faster-whisper-large-v3-ja'
# model_path = 'large-v3-turbo'
# model = WhisperModel(model_path, device='cuda', compute_type='float16', download_root='.')
model_path = (ROOT / 'model_zoo/faster-whisper-large-v3').as_posix()
# model = WhisperModel(model_path, device='cuda', compute_type='float16', local_files_only=True)
# or run on GPU with INT8
model = WhisperModel(model_path, device='cuda', compute_type='int8_float16', local_files_only=True)
# or run on CPU with INT8
# model = WhisperModel(model_path, device='cpu', compute_type='int8')

def run(path: Path):
  segments, info = model.transcribe(path.as_posix(), language='ja', beam_size=5)
  resolver.reset_input(title=path.stem)
  f = None
  try:
    f = path.with_suffix('.ass').open('w', encoding='utf-8')
    pprint = lambda x: (print(x, end=''), f.write(x))
    pprint(resolver.format_pre())
    pprint(f'; faster_whisper: Detected language \'{info.language}\' with probability {info.language_probability}\n')
    for segment in tqdm(segments):
      # pprint(f'[{segment.start}s -> {segment.end}s] {segment.text}\n')
      st = resolver.format_seconds(segment.start)
      et = resolver.format_seconds(segment.end)
      f.write(resolver.format_line(st, et, segment.text))
    pprint(resolver.format_post())
  except Exception:
    err_log = f'error_{path.stem}.log'
    traceback.print_exc(file=open(err_log, 'w', encoding='utf-8'))
    raise
  finally:
    if f is not None:
      f.close()


if __name__ == '__main__':
  for p in paths:
    print(f'Processing {p}...')
    run(p)
