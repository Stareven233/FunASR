'''
[whisper 输入转 srt]
python scripts/subtitles.py -t whisper -s srt -p xxx.txt

[funasr-nano 输入转 srt]
python scripts/subtitles.py -t funasr-nano -s srt -p xxx.txt

[funasr-nano 输入转 ass]
python scripts/subtitles.py -t funasr-nano -s ass -p xxx.txt

uv run scripts/subtitles.py --type funasr-nano --subtitle-type srt --input "{'key': 'output', 'text': '首先解压下载的压缩包。 然后', 'text_tn': '首先解压下载的压缩包 sil 然后', 'label': 'null', 'ctc_text': '首先解压下载的压缩包  然后', 'ctc_timestamps': [{'token': '首', 'start_time': 1.26, 'end_time': 1.32, 'score': 0.998}, {'token': '先', 'start_time': 1.44, 'end_time': 1.5, 'score': 1.0}, {'token': '解', 'start_time': 1.62, 'end_time': 1.68, 'score': 0.999}, {'token': '压', 'start_time': 1.8, 'end_time': 1.86, 'score': 0.991}, {'token': '下', 'start_time': 2.04, 'end_time': 2.1, 'score': 1.0}, {'token': '载', 'start_time': 2.16, 'end_time': 2.22, 'score': 1.0}, {'token': '的', 'start_time': 2.28, 'end_time': 2.34, 'score': 0.64}, {'token': '压', 'start_time': 2.46, 'end_time': 2.52, 'score': 0.993}, {'token': '缩', 'start_time': 2.64, 'end_time': 2.7, 'score': 0.986}, {'token': '包', 'start_time': 2.82, 'end_time': 2.88, 'score': 0.999}, {'token': ' ', 'start_time': 3.18, 'end_time': 3.3, 'score': 0.946}, {'token': ' ', 'start_time': 8.52, 'end_time': 8.58, 'score': 0.706}, {'token': '然', 'start_time': 9.0, 'end_time': 9.06, 'score': 0.997}, {'token': '后', 'start_time': 9.12, 'end_time': 9.18, 'score': 0.991}]}"
'''
import ast
import re
import argparse
from typing import Iterator
from pathlib import Path


class SRT_Resolver:
  def __init__(
    self,
    content: str,
    title='',
    input_type='whisper',
    max_sentence_duration: float = 3.5,
    max_sentence_length: int = 24,
  ) -> None:
    self.title = title
    self.input_type = input_type
    # 超出改间隔、且超出限制长度将被截断
    self.max_sentence_duration = max_sentence_duration
    self.max_sentence_length = max_sentence_length
    self.time_format = r'{:02}:{:02}:{:02},{:03} --> {:02}:{:02}:{:02},{:03}'
    self.timelines: list[tuple]
    self.resolve_pattern = re.compile(r'\[(\d+\.\d+)s\s->\s(\d+\.\d+)s\]\s+(.+)')
    self.reset_input(content)

  def reset_input(self, content='', title=None):
    '''
      whisper_output: ([28.22s -> 31.04s] エクスプローション!, ...)
      timelines: [(start_seconds, end_seconds, text), ...]
    '''
    if content != '':
      if self.input_type == 'funasr-nano':
        self.timelines = self.resolve_funasr_nano(content)
      else:
        self.timelines = self.resolve_whisper(content)
    if title is not None:
      self.title = title

  def resolve_whisper(self, content: str):
    timelines = []
    for line in content.split('\n'):
      if line == '':
        continue
      m = self.resolve_pattern.search(line)
      if m is None:
        continue
      timelines.append(m.groups())
    return timelines

  def split_funasr_nano_sentences(self, text: str, timestamps: list[dict]):
    sentence_endings = set('。！？!?；;')
    split_punctuation = set('，,。.')
    punctuation = sentence_endings | set('，、：“”‘’（）()《》【】…,.')
    tokens = [token for token in timestamps if token.get('token', '').strip()]
    token_index = 0
    sentence_chars = []
    sentence_start = None
    sentence_end = None
    timelines = []

    def flush_sentence():
      nonlocal sentence_chars, sentence_start, sentence_end
      sentence = ''.join(sentence_chars).strip()
      if sentence and sentence_start is not None and sentence_end is not None:
        timelines.append((sentence_start, sentence_end, sentence))
      sentence_chars = []
      sentence_start = None
      sentence_end = None

    for ch in text:
      sentence_chars.append(ch)
      if (not ch.isspace()) and (ch not in punctuation):
        if token_index >= len(tokens):
          continue
        token = tokens[token_index]
        token_index += 1
        if sentence_start is None:
          sentence_start = token['start_time']
        sentence_end = token['end_time']

      sentence = ''.join(sentence_chars).strip()
      sentence_duration = 0 if sentence_start is None or sentence_end is None else sentence_end - sentence_start
      if ch in sentence_endings:
        flush_sentence()
      elif (
        ch in split_punctuation
        and sentence_start is not None
        and sentence_end is not None
        and sentence_duration > self.max_sentence_duration
        and len(sentence) > self.max_sentence_length
      ):
        flush_sentence()

    flush_sentence()
    return timelines

  def resolve_funasr_nano(self, content: str):
    item = ast.literal_eval(content)
    timelines = []
    text = item.get('text', '')
    # 启用VAD时有timestamps，会考虑分片的时间连续，此时ctc_timestamps是错的，分片间时间会重置
    timestamps = item.get('timestamps', None) or item.get('ctc_timestamps', [])
    timelines.extend(self.split_funasr_nano_sentences(text, timestamps))
    return timelines

  @staticmethod
  def format_seconds(second):
    second = float(second)
    total_ms = round(second * 1000)
    total_seconds, ms = divmod(total_ms, 1000)
    h, remainder = divmod(total_seconds, 3600)
    m, s = divmod(remainder, 60)
    return h, m, s, ms

  def format_pre(self):
    return ''

  def format_line(self, st, et, text):
    return f'{self.time_format.format(*st, *et)}\n{text}\n\n'

  def format_post(self):
    return ''

  def __call__(self):
    yield self.format_pre()
    for (st, et, text) in self.timelines:
      st = self.format_seconds(st)
      et = self.format_seconds(et)
      yield self.format_line(st, et, text)
    yield self.format_post()


class ASS_Resolver(SRT_Resolver):
  def __init__(
    self,
    content: str,
    title='',
    input_type='whisper',
    max_sentence_duration: float = 3.5,
    max_sentence_length: int = 24,
  ) -> None:
    super().__init__(content, title, input_type, max_sentence_duration, max_sentence_length)
    self.time_format = r'{:01}:{:02}:{:02}.{:02},{:01}:{:02}:{:02}.{:02}'

  def format_pre(self):
    return f'''[Script Info]
      Title: {self.title}
      Original Script: Noe with {self.__class__.__name__}
      ScriptType: v4.00+
      WrapStyle: 0
      PlayResX: 1920
      PlayResY: 1080

      [V4+ Styles]
      Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
      Style: DS, 微软雅黑, 72, &H00FFFFFF, &H00000000, &H00000000,&H0050C3F9,0,0,0,0,100,100,0,0,1,3,0,2,10,10,25,1

      [Events]
      Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
    '''.replace('  ', '')

  def format_line(self, st, et, text):
    time = f'{self.time_format.format(*st, *et)}'
    return f'Dialogue: 0,{time},DS,,0,0,0,,{text}\n'


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--path', '-p', help='input transcript file path', type=Path, default=None)
  parser.add_argument('--input', '-i', help='input transcript string', type=str, default=None)
  parser.add_argument('--type', '-t', help='input transcript type', choices=['whisper', 'funasr-nano'], type=str, default='whisper')
  parser.add_argument('--subtitle-type', '-s', help='subtitle output type', choices=['srt', 'ass'], type=str, default='srt')
  args = parser.parse_args()
  return args


def run(
  content: str,
  input_type: str,
  subtitle_type: str,
  stream: bool = False,
) -> str | Iterator[str]:
  resolver_cls = dict(srt=SRT_Resolver, ass=ASS_Resolver).get(subtitle_type.lower(), SRT_Resolver)
  resolver = resolver_cls(
    content,
    input_type=input_type.lower(),
    max_sentence_duration=3.5,
    max_sentence_length=24,
  )
  chunks = resolver()
  if stream:
    return chunks
  return ''.join(chunks)


if __name__ == '__main__':
  from tqdm import tqdm

  args = parse_args()
  path: Path = args.path
  input_type = args.type.lower()
  subtitle_type = args.subtitle_type.lower()
  content: str = args.input
  if path is not None:
    with path.open('r', encoding='utf-8') as f:
      content = f.read()
  content = content or r'''
    [0.68s -> 6.0206000000000005s] 黒より黒く、闇より暗く漆黒に、我が真紅の今後を望みたもう。
    [6.8s -> 12.5s] 覚醒の時来たれり、無病の境界に落ちし断り、無行の歪みとなりて現出せよ!
    [13.14s -> 20.12s] 踊れ、踊れ、踊れ!我が力の本流に望むは崩壊なり、並ぶ者なき崩壊なり!
    [20.64s -> 23.32s] 万象ひたしく怪人に来し、深淵より来たれ!
    [23.86s -> 28.22s] 終焉の王国の地に力の根源を引いてくせし者を我が前に滑べよ!
    [28.22s -> 31.04s] エクスプローション!
    [32.14s -> 32.36s] ドン!
  '''

  if path is None:
    subtitle = run(content, input_type=input_type, subtitle_type=subtitle_type)
    print(subtitle, end='')
  else:
    output_path = path.with_suffix(f'.{subtitle_type}')
    with output_path.open('w', encoding='utf-8') as f:
      for chunk in tqdm(run(content, input_type=input_type, subtitle_type=subtitle_type, stream=True)):
        f.write(chunk)
    print(f'{f.name} saved')
