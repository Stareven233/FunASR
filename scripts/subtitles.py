import re


class SRT_Resolver:
  def __init__(self, whisper_output: str, title='') -> None:
    self.title = title
    self.time_format = r'{:02}:{:02}:{:02},{:03} --> {:02}:{:02}:{:02},{:03}'
    self.timelines: list[tuple]
    self.resolve_pattern = re.compile(r'\[(\d+\.\d+)s\s->\s(\d+\.\d+)s\]\s+(.+)')
    self.reset_input(whisper_output)

  def reset_input(self, whisper_output='', title=None):
    '''
      whisper_output: ([28.22s -> 31.04s] エクスプローション!, ...)
      timelines: [(start_seconds, end_seconds, text), ...]
    '''
    if whisper_output != '':
      self.timelines = []
      for line in whisper_output.split('\n'):
        if line == '':
          continue
        m = self.resolve_pattern.search(line)
        if m is None:
          continue
        self.timelines.append(m.groups())
    if title is not None:
      self.title = title

  @staticmethod
  def format_seconds(second):
    if not isinstance(second, str):
      second = str(second)
    # 将秒数分割成整数部分和小数部分
    s, ms = second.split('.')
    if len(ms) > 3:
      ms = round(float(f'{ms[:3]}.{ms[3:]}'), 0)
    # 计算小时、分钟和秒
    h, remainder = divmod(int(s), 3600)
    m, s = divmod(remainder, 60)
    ms = int(ms)
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
  def __init__(self, whisper_output: str, title='') -> None:
    super().__init__(whisper_output, title)
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


if __name__ == '__main__':
  import argparse
  from pathlib import Path
  from tqdm import tqdm

  parser = argparse.ArgumentParser()
  parser.add_argument("--path", '-p', help="input whisper output file path", type=Path, default=None)
  parser.add_argument("--input", '-i', help="input whisper output string", type=str, default=None)
  parser.add_argument("--type", '-t', help="subtitle type", type=str, default='SRT')
  args = parser.parse_args()

  path: Path = args.path
  stype = args.type.lower()
  title: str = 'no-title'
  content: str = args.input
  if path is not None:
    title = path.stem
    with path.open('r', encoding='utf-8') as f:
      content = f.read()
  content = content or r"""
    [0.68s -> 6.0206000000000005s] 黒より黒く、闇より暗く漆黒に、我が真紅の今後を望みたもう。
    [6.8s -> 12.5s] 覚醒の時来たれり、無病の境界に落ちし断り、無行の歪みとなりて現出せよ!
    [13.14s -> 20.12s] 踊れ、踊れ、踊れ!我が力の本流に望むは崩壊なり、並ぶ者なき崩壊なり!
    [20.64s -> 23.32s] 万象ひたしく怪人に来し、深淵より来たれ!
    [23.86s -> 28.22s] 終焉の王国の地に力の根源を引いてくせし者を我が前に滑べよ!
    [28.22s -> 31.04s] エクスプローション!
    [32.14s -> 32.36s] ドン!
  """

  r = dict(srt=SRT_Resolver, ass=ASS_Resolver).get(stype, SRT_Resolver)
  r = r(content)
  if path is None:
    for t in r():
      print(t, end='')
  else:
    f = path.with_suffix(f'.{stype}').open('w', encoding='utf-8')
    for t in tqdm(r()):
      f.write(t)
    f.close()
    print(f'{f.name} saved')
