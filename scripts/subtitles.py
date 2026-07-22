'''
[whisper 输入转 srt]
python scripts/subtitles.py -t whisper -s srt -p xxx.txt

[funasr-nano 输入转 srt]
python scripts/subtitles.py -t funasr-nano -s srt -p xxx.txt

[funasr-nano 输入转 ass]
python scripts/subtitles.py -t funasr-nano -s ass -p xxx.txt

uv run scripts/subtitles.py --type funasr-nano --subtitle-type srt --input "{'key': 'output', 'text': '首先解压下载的压缩包。 然后', 'text_tn': '首先解压下载的压缩包 sil 然后', 'label': 'null', 'ctc_text': '首先解压下载的压缩包  然后', 'ctc_timestamps': [{'token': '首', 'start_time': 1.26, 'end_time': 1.32, 'score': 0.998}, {'token': '先', 'start_time': 1.44, 'end_time': 1.5, 'score': 1.0}, {'token': '解', 'start_time': 1.62, 'end_time': 1.68, 'score': 0.999}, {'token': '压', 'start_time': 1.8, 'end_time': 1.86, 'score': 0.991}, {'token': '下', 'start_time': 2.04, 'end_time': 2.1, 'score': 1.0}, {'token': '载', 'start_time': 2.16, 'end_time': 2.22, 'score': 1.0}, {'token': '的', 'start_time': 2.28, 'end_time': 2.34, 'score': 0.64}, {'token': '压', 'start_time': 2.46, 'end_time': 2.52, 'score': 0.993}, {'token': '缩', 'start_time': 2.64, 'end_time': 2.7, 'score': 0.986}, {'token': '包', 'start_time': 2.82, 'end_time': 2.88, 'score': 0.999}, {'token': ' ', 'start_time': 3.18, 'end_time': 3.3, 'score': 0.946}, {'token': ' ', 'start_time': 8.52, 'end_time': 8.58, 'score': 0.706}, {'token': '然', 'start_time': 9.0, 'end_time': 9.06, 'score': 0.997}, {'token': '后', 'start_time': 9.12, 'end_time': 9.18, 'score': 0.991}]}"
uv run scripts/subtitles.py --type funasr-nano --subtitle-type srt --path "D:/Document/Video/leafflow/vocal/output.json" --save
'''
import ast
import re
import argparse
from typing import Iterator
from pathlib import Path


def clean_text(text: str) -> str:
  """去除模型输出中的 <|tag|> 标记"""
  return re.sub(r'<\|[^|]*\|>', '', text or '').strip()


def timestamp_bounds_ms(result: dict) -> tuple[int, int] | None:
  """
  从 token 级 timestamps 推导整段起止时间（毫秒）。
  参考 examples/subtitle/generate_subtitle.py，用于 sentence_info 缺失时的回退。
  """
  bounds = []
  for key in ('timestamp', 'timestamps', 'ctc_timestamps'):
    for ts in result.get(key, []) or []:
      if isinstance(ts, dict):
        start = ts.get('start_time', ts.get('start'))
        end = ts.get('end_time', ts.get('end'))
        if start is None or end is None:
          continue
        # token 时间戳多为秒；start/end 若已是毫秒级大数则不再 *1000
        start_f, end_f = float(start), float(end)
        if start_f < 1000 and end_f < 10000:
          start_ms = int(start_f * 1000)
          end_ms = int(end_f * 1000)
        else:
          start_ms, end_ms = int(start_f), int(end_f)
      elif isinstance(ts, (list, tuple)) and len(ts) >= 2:
        start_ms = int(ts[0])
        end_ms = int(ts[1])
      else:
        continue
      if end_ms > start_ms:
        bounds.append((start_ms, end_ms))
  if not bounds:
    return None
  return min(start for start, _ in bounds), max(end for _, end in bounds)


class SRT_Resolver:
  def __init__(
    self,
    content: str | dict,
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

  def reset_input(self, content: str | dict = '', title=None):
    '''
      whisper_output: ([28.22s -> 31.04s] エクスプローション!, ...)
      timelines: [(start_seconds, end_seconds, text), ...]
    '''
    if content != '' and content is not None:
      if self.input_type == 'funasr-nano':
        self.timelines = self.resolve_funasr_nano(content)
      else:
        self.timelines = self.resolve_whisper(content if isinstance(content, str) else str(content))
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

    if not timestamps:
      return [(0, 0, text)]

    timelines = []
    group = []

    def _start_time(tokens):
      for t in tokens:
        tt = t.get('token', '').strip()
        if tt and tt not in sentence_endings | split_punctuation:
          return t.get('start_time')
      return tokens[0].get('start_time', 0) if tokens else 0

    def _end_time(tokens):
      for t in reversed(tokens):
        tt = t.get('token', '').strip()
        if tt and tt not in sentence_endings | split_punctuation:
          return t.get('end_time')
      return tokens[-1].get('end_time', 0) if tokens else 0

    def _flush():
      nonlocal group
      if not group:
        return
      st = _start_time(group)
      et = _end_time(group)
      s = ''.join(t.get('token', '') for t in group).strip()
      if s:
        timelines.append((st, et, s))
      group = []

    for token in timestamps:
      group.append(token)
      t_text = token.get('token', '').strip()

      if t_text in sentence_endings:
        _flush()
      elif t_text in split_punctuation and len(group) > 1:
        st = _start_time(group[:-1])
        et = _end_time(group[:-1])
        if st is not None and et is not None:
          if (et - st) > self.max_sentence_duration:
            s = ''.join(t.get('token', '') for t in group).strip()
            if len(s) > self.max_sentence_length:
              _flush()

    _flush()
    return timelines

  def resolve_funasr_nano(self, content: str | dict):
    """
    解析 Fun-ASR-Nano 结果。
    优先级：
      1. sentence_info（sentence_timestamp=True 时的分句结果，start/end 为毫秒）
      2. timestamps / ctc_timestamps 按标点分句（秒）
      3. 整段 text + timestamp_bounds_ms 回退，避免 VAD 偶发空 sentence_info 时静默丢字幕
    """
    item = content if isinstance(content, dict) else ast.literal_eval(content)

    # 1) 优先 sentence_info
    timelines = []
    for seg in item.get('sentence_info', []) or []:
      text = clean_text(seg.get('sentence') or seg.get('text', ''))
      start_ms = int(seg.get('start', 0) or 0)
      end_ms = int(seg.get('end', 0) or 0)
      if text and end_ms > start_ms:
        timelines.append((start_ms / 1000.0, end_ms / 1000.0, text))
    if timelines:
      return timelines

    # 2) token 级 timestamps 分句
    # 启用 VAD 时有 timestamps，会考虑分片的时间连续；此时 ctc_timestamps 分片间时间会重置，优先 timestamps
    text = item.get('text', '')
    timestamps = item.get('timestamps') or item.get('ctc_timestamps') or []
    if timestamps:
      return self.split_funasr_nano_sentences(text, timestamps)

    # 3) 整段回退，避免 sentence_info 为空时丢失字幕
    text = clean_text(text)
    if not text:
      return []
    bounds = timestamp_bounds_ms(item)
    if bounds:
      st_ms, et_ms = bounds
      return [(st_ms / 1000.0, et_ms / 1000.0, text)]
    return [(0.0, 0.0, text)]

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

  def format_line(self, i, st, et, text):
    return f'{i}\n{self.time_format.format(*st, *et)}\n{text}\n\n'

  def format_post(self):
    return ''

  def __call__(self):
    yield self.format_pre()
    for i, (st, et, text) in enumerate(self.timelines):
      st = self.format_seconds(st)
      et = self.format_seconds(et)
      yield self.format_line(i, st, et, text)
    yield self.format_post()


class ASS_Resolver(SRT_Resolver):
  def __init__(
    self,
    content: str | dict,
    title='',
    input_type='whisper',
    max_sentence_duration: float = 3.5,
    max_sentence_length: int = 24,
  ) -> None:
    super().__init__(content, title, input_type, max_sentence_duration, max_sentence_length)
    self.time_format = r'{:01}:{:02}:{:02}.{:02},{:01}:{:02}:{:02}.{:02}'

  def format_pre(self):
    # title 可由 CLI --title 覆盖；未指定时用空 Title，避免硬编码作者信息
    title = self.title or 'FunASR'
    return f'''[Script Info]
      Title: {title}
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

  def format_line(self, i, st, et, text):
    # ASS 百分秒只需两位
    st = (*st[:3], st[3] // 10)
    et = (*et[:3], et[3] // 10)
    time = f'{self.time_format.format(*st, *et)}'
    return f'Dialogue: 0,{time},DS,,0,0,0,,{text}\n'


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--path', '-p', help='input transcript file path', type=Path, default=None)
  parser.add_argument('--input', '-i', help='input transcript string', type=str, default=None)
  parser.add_argument('--type', '-t', help='input transcript type', choices=['whisper', 'funasr-nano'], type=str, default='whisper')
  parser.add_argument('--subtitle-type', '-s', help='subtitle output type', choices=['srt', 'ass'], type=str, default='srt')
  parser.add_argument('--title', help='ASS/SRT title metadata', type=str, default='')
  parser.add_argument('--save', help='save', action='store_true')
  args = parser.parse_args()
  return args


def run(
  content: str | dict,
  input_type: str,
  subtitle_type: str,
  stream: bool = False,
  title: str = '',
) -> str | Iterator[str]:
  resolver_cls = dict(srt=SRT_Resolver, ass=ASS_Resolver).get(subtitle_type.lower(), SRT_Resolver)
  resolver = resolver_cls(
    content,
    title=title,
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
      raw = f.read()
    # json 文件优先按 JSON 解析为 dict，避免 literal_eval 对 true/null 的差异
    if path.suffix.lower() == '.json':
      import json

      content = json.loads(raw)
    else:
      content = raw
  content = content or r'''
    [0.68s -> 6.0206000000000005s] 黒より黒く、闇より暗く漆黒に、我が真紅の今後を望みたもう。
    [6.8s -> 12.5s] 覚醒の時来たれり、無病の境界に落ちし断り、無行の歪みとなりて現出せよ!
    [13.14s -> 20.12s] 踊れ、踊れ、踊れ!我が力の本流に望むは崩壊なり、並ぶ者なき崩壊なり!
    [20.64s -> 23.32s] 万象ひたしく怪人に来し、深淵より来たれ!
    [23.86s -> 28.22s] 終焉の王国の地に力の根源を引いてくせし者を我が前に滑べよ!
    [28.22s -> 31.04s] エクスプローション!
    [32.14s -> 32.36s] ドン!
  '''

  subtitle = run(
    content,
    input_type=input_type,
    subtitle_type=subtitle_type,
    stream=True,
    title=args.title,
  )
  if path is not None and args.save:
    output_path = path.with_suffix(f'.{subtitle_type}')
    with output_path.open('w', encoding='utf-8') as f:
      for chunk in tqdm(subtitle):
        f.write(chunk)
    print(f'{f.name} saved')
  else:
    for s in subtitle:
      print(s, end='')
