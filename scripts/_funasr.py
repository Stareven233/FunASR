'''https://github.com/modelscope/FunASR/blob/main/README_zh.md

model hub: https://www.modelscope.cn/models?page=1&tasks=auto-speech-recognition  <— dowload with git lfs

uv pip install torch==1.13.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
uv pip install -e ./
uv pip install numpy<2

cd D:\Code\projects\FunASR
$audio = "D:/Document/Audio/!raw/篠宮ゆり.aac"
$audio = "D:/Document/ai-sings/Ending Note/Ending Note 門谷純_Vocals_vocals.flac"
uv run scripts/_funasr.py -m sensevoice -i $audio
'''


import sys

from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

sys.path.append('.')
from scripts import ROOT, parser


model_mapping = dict(
    sensevoice='SenseVoiceSmall',
    # 对whisper支持不全，一坨！
    whisper='Whisper-large-v3-turbo',
)

args = parser.parse_args()
mname = args.model_name
if not mname:
    mname = 'sensevoice'
assert mname in model_mapping
vad_model_dir = ROOT / 'model_zoo/speech_fsmn_vad_zh-cn-16k-common-pytorch'

model = ROOT / f'model_zoo/{model_mapping[mname]}'
model = AutoModel(
    model=model,
    vad_model=vad_model_dir,
    vad_kwargs=vad_model_dir and {'max_single_segment_time': 30000},
    device='cuda:0',
)
DecodingOptions = {
    'task': 'transcribe',
    'language': None,
    'beam_size': None,
    'fp16': True,
    'without_timestamps': False,
    'prompt': None,
}


def run(path):
    res = model.generate(
        input=path.resolve().as_posix(),
        # DecodingOptions=DecodingOptions,
        # input=f'D:/Document/Audio/rie.aac',
        # input=f'https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav',
        cache={},
        language='auto',  # 'zn', 'en', 'yue', 'ja', 'ko', 'nospeech'
        use_itn=True,
        batch_size_s=0 if mname == 'whisper' else 60,
        merge_vad=vad_model_dir and True,
        merge_length_s=15,
    )
    print(res)  # [{'key': 'en', 'text': '<|en|><|NEUTRAL|><|Speech|><|withitn|>The tribal chieftain called for the boy and presented him with 50 pieces of gold.'}]
    text = rich_transcription_postprocess(res[0]['text'])
    print(text)


if __name__ == '__main__':
    for i in args.input:
        run(i)

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
