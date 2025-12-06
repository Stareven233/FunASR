import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
parser = argparse.ArgumentParser()
parser.add_argument(
    '-i',
    '--input',
    type=Path,
    action='append',
    required=True,
    help='path to the input audio (whisper only accept wav in this project)',
)
parser.add_argument(
    '-m',
    '--model_name',
    type=str,
    required=False,
    help='path to the model name',
)