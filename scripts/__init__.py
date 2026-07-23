import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent


def parse_transcribe_args():
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
  parser.add_argument(
    '-s',
    '--subtitle-type',
    type=str,
    # choices=['srt', 'ass'],
    required=False,
    default=None,
    help='subtitle output type; when omitted, keep original model output. allow union [srt, ass, srt|raw, ass|raw]',
  )
  parser.add_argument(
    '-l',
    '--language',
    type=str,
    default='auto',
    help="recognition language (default: auto). e.g. auto / 中文 / ja / en; mix-language video should keep auto",
  )
  parser.add_argument(
    '--vad',
    action=argparse.BooleanOptionalAction,
    default=True,
    help='enable VAD for long audio (default: on). short clips can use --no-vad to skip VAD',
  )
  parser.add_argument(
    '--title',
    type=str,
    default='',
    help='subtitle metadata title (mainly for ASS Script Info Title)',
  )
  parser.add_argument(
    '--batch-size',
    type=int,
    default=8,
    help='ASR sample batch size (default: 8). With VAD, keep this high so packed segments are not re-sliced to 1; try 8/16',
  )
  parser.add_argument(
    '--batch-size-s',
    type=int,
    default=60,
    help='VAD dynamic-batch total speech duration in seconds (default: 60). Larger → fewer ASR calls, faster',
  )
  parser.add_argument(
    '--save-to-file',
    action='store_true',
    default=True,
    help='save processed result to file instead of printing it',
  )
  return parser.parse_args()


# 以下代码为了解决报错：
# Traceback (most recent call last):
#   File "D:\Code\projects\FunASR\scripts\_funasr.py", line 101, in <module>
#     main(args)
#   File "D:\Code\projects\FunASR\scripts\_funasr.py", line 85, in main
#     model = AutoModel(
#             ^^^^^^^^^^
#   File "D:\Code\projects\FunASR\funasr\auto\auto_model.py", line 141, in __init__
#     model, kwargs = self.build_model(**kwargs)
#                     ^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "D:\Code\projects\FunASR\funasr\auto\auto_model.py", line 292, in build_model
#     model = model_class(**model_conf)
#             ^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "D:\Code\projects\FunASR\funasr/models/fun_asr_nano\model.py", line 61, in __init__
#     audio_encoder = encoder_class(input_size=input_size, **audio_encoder_conf)
#                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# TypeError: 'NoneType' object is not callable

from funasr.register import tables
from funasr.datasets.audio_datasets.datasets import AudioDataset
tables.register('dataset_classes', 'AudioDataset')(AudioDataset)
from funasr.datasets.audio_datasets.datasets import AudioDatasetHotword
tables.register('dataset_classes', 'AudioDatasetHotword')(AudioDatasetHotword)
from funasr.datasets.audio_datasets.espnet_samplers import EspnetStyleBatchSampler_fn
tables.register('batch_sampler_classes', 'EspnetStyleBatchSampler')(EspnetStyleBatchSampler_fn)
from funasr.datasets.audio_datasets.index_ds import IndexDSJsonlRankFull
tables.register('index_ds_classes', 'IndexDSJsonlRankSplit')(IndexDSJsonlRankFull)
from funasr.datasets.audio_datasets.index_ds import IndexDSJsonlRankFull
tables.register('index_ds_classes', 'IndexDSJsonlRankFull')(IndexDSJsonlRankFull)
from funasr.datasets.audio_datasets.index_ds import IndexDSJsonlRankFull
tables.register('index_ds_classes', 'IndexDSJsonl')(IndexDSJsonlRankFull)
from funasr.datasets.audio_datasets.preprocessor import SpeechPreprocessSpeedPerturb
tables.register('preprocessor_classes', 'SpeechPreprocessSpeedPerturb')(SpeechPreprocessSpeedPerturb)
from funasr.datasets.audio_datasets.preprocessor import TextPreprocessSegDict
tables.register('preprocessor_classes', 'TextPreprocessSegDict')(TextPreprocessSegDict)
from funasr.datasets.audio_datasets.samplers import CustomDistributedBatchSampler_fn
tables.register('batch_sampler_classes', 'RankFullLocalShuffleDynamicBatchSampler')(CustomDistributedBatchSampler_fn)
from funasr.datasets.audio_datasets.samplers import CustomDistributedBatchSampler_fn
tables.register('batch_sampler_classes', 'RankFullLocalShuffleBatchSampler')(CustomDistributedBatchSampler_fn)
from funasr.datasets.audio_datasets.samplers import CustomDistributedBatchSampler_fn
tables.register('batch_sampler_classes', 'DynamicBatchLocalShuffleSampler')(CustomDistributedBatchSampler_fn)
from funasr.datasets.audio_datasets.samplers import CustomDistributedBatchSampler_fn
tables.register('batch_sampler_classes', 'CustomDistributedDynamicBatchSampler')(CustomDistributedBatchSampler_fn)
from funasr.datasets.audio_datasets.samplers import CustomDistributedBatchSampler_fn
tables.register('batch_sampler_classes', 'CustomDistributedBatchSampler')(CustomDistributedBatchSampler_fn)
from funasr.datasets.audio_datasets.samplers import CustomDistributedBatchSampler_fn
tables.register('batch_sampler_classes', 'BatchSampler')(CustomDistributedBatchSampler_fn)
from funasr.datasets.dataloader_entry import DataloaderMapStyle
tables.register('dataloader_classes', 'DataloaderMapStyle')(DataloaderMapStyle)
from funasr.datasets.dataloader_entry import DataloaderIterable
tables.register('dataloader_classes', 'DataloaderIterable')(DataloaderIterable)
from funasr.datasets.kws_datasets.datasets import KwsMTDataset
tables.register('dataset_classes', 'KwsMTDataset')(KwsMTDataset)
from funasr.datasets.llm_datasets.datasets import AudioLLMNARDataset
tables.register('dataset_classes', 'AudioLLMNARDataset')(AudioLLMNARDataset)
from funasr.datasets.llm_datasets.datasets import AudioLLMDataset
tables.register('dataset_classes', 'AudioLLMDataset')(AudioLLMDataset)
from funasr.datasets.llm_datasets.datasets import AudioLLMARDataset
tables.register('dataset_classes', 'AudioLLMARDataset')(AudioLLMARDataset)
from funasr.datasets.llm_datasets.preprocessor import TextPreprocessRemovePunctuation
tables.register('preprocessor_classes', 'TextPreprocessRemovePunctuation')(TextPreprocessRemovePunctuation)
from funasr.datasets.llm_datasets_qwenaudio.datasets import AudioLLMQwenAudioDataset
tables.register('dataset_classes', 'AudioLLMQwenAudioDataset')(AudioLLMQwenAudioDataset)
from funasr.datasets.llm_datasets_vicuna.datasets import AudioLLMVicunaDataset
tables.register('dataset_classes', 'AudioLLMVicunaDataset')(AudioLLMVicunaDataset)
from funasr.datasets.openai_datasets.datasets import OpenAIDataset
tables.register('dataset_classes', 'OpenAIDataset')(OpenAIDataset)
from funasr.datasets.openai_datasets.datasets import OpenAIDatasetMultiTurn
tables.register('dataset_classes', 'OpenAIDatasetMultiTurn')(OpenAIDatasetMultiTurn)
from funasr.datasets.openai_datasets.index_ds import OpenAIIndexDSJsonl
tables.register('index_ds_classes', 'OpenAIIndexDSJsonl')(OpenAIIndexDSJsonl)
from funasr.datasets.sense_voice_datasets.datasets import SenseVoiceDataset
tables.register('dataset_classes', 'SenseVoiceDataset')(SenseVoiceDataset)
from funasr.datasets.sense_voice_datasets.datasets import SenseVoiceCTCDataset
tables.register('dataset_classes', 'SenseVoiceCTCDataset')(SenseVoiceCTCDataset)
from funasr.frontends.wav_frontend import WavFrontend
tables.register('frontend_classes', 'WavFrontend')(WavFrontend)
from funasr.frontends.wav_frontend import WavFrontend
tables.register('frontend_classes', 'wav_frontend')(WavFrontend)
from funasr.frontends.wav_frontend import WavFrontendOnline
tables.register('frontend_classes', 'WavFrontendOnline')(WavFrontendOnline)
from funasr.frontends.whisper_frontend import WhisperFrontend
tables.register('frontend_classes', 'WhisperFrontend')(WhisperFrontend)
from funasr.models.transducer.joint_network import JointNetwork
tables.register('joint_network_classes', 'joint_network')(JointNetwork)
from funasr.models.transducer.model import Transducer
tables.register('model_classes', 'Transducer')(Transducer)
from funasr.models.bat.model import BAT
tables.register('model_classes', 'BAT')(BAT)
from funasr.models.bicif_paraformer.cif_predictor import CifPredictorV3
tables.register('predictor_classes', 'CifPredictorV3')(CifPredictorV3)
from funasr.models.bicif_paraformer.cif_predictor import CifPredictorV3Export
tables.register('predictor_classes', 'CifPredictorV3Export')(CifPredictorV3Export)
from funasr.models.paraformer.cif_predictor import CifPredictor
tables.register('predictor_classes', 'CifPredictor')(CifPredictor)
from funasr.models.paraformer.cif_predictor import CifPredictorV2
tables.register('predictor_classes', 'CifPredictorV2')(CifPredictorV2)
from funasr.models.paraformer.cif_predictor import CifPredictorV2Export
tables.register('predictor_classes', 'CifPredictorV2Export')(CifPredictorV2Export)
from funasr.models.paraformer.model import Paraformer
tables.register('model_classes', 'Paraformer')(Paraformer)
from funasr.models.bicif_paraformer.model import BiCifParaformer
tables.register('model_classes', 'BiCifParaformer')(BiCifParaformer)
from funasr.models.branchformer.encoder import BranchformerEncoder
tables.register('encoder_classes', 'BranchformerEncoder')(BranchformerEncoder)
from funasr.models.transformer.model import Transformer
tables.register('model_classes', 'Transformer')(Transformer)
from funasr.models.branchformer.model import Branchformer
tables.register('model_classes', 'Branchformer')(Branchformer)
from funasr.models.campplus.model import CAMPPlus
tables.register('model_classes', 'CAMPPlus')(CAMPPlus)
from funasr.models.conformer.encoder import ConformerEncoder
tables.register('encoder_classes', 'ConformerEncoder')(ConformerEncoder)
from funasr.models.conformer.encoder import ConformerChunkEncoder
tables.register('encoder_classes', 'ChunkConformerEncoder')(ConformerChunkEncoder)
from funasr.models.conformer.model import Conformer
tables.register('model_classes', 'Conformer')(Conformer)
from funasr.models.conformer_rwkv.decoder import TransformerRWKVDecoder
tables.register('decoder_classes', 'TransformerRWKVDecoder')(TransformerRWKVDecoder)
from funasr.models.conformer_rwkv.model import Conformer
tables.register('model_classes', 'Conformer')(Conformer)
from funasr.models.transformer.decoder import TransformerDecoder
tables.register('decoder_classes', 'TransformerDecoder')(TransformerDecoder)
from funasr.models.transformer.decoder import LightweightConvolutionTransformerDecoder
tables.register('decoder_classes', 'LightweightConvolutionTransformerDecoder')(LightweightConvolutionTransformerDecoder)
from funasr.models.transformer.decoder import LightweightConvolution2DTransformerDecoder
tables.register('decoder_classes', 'LightweightConvolution2DTransformerDecoder')(LightweightConvolution2DTransformerDecoder)
from funasr.models.transformer.decoder import DynamicConvolutionTransformerDecoder
tables.register('decoder_classes', 'DynamicConvolutionTransformerDecoder')(DynamicConvolutionTransformerDecoder)
from funasr.models.transformer.decoder import DynamicConvolution2DTransformerDecoder
tables.register('decoder_classes', 'DynamicConvolution2DTransformerDecoder')(DynamicConvolution2DTransformerDecoder)
from funasr.models.paraformer.decoder import ParaformerSANMDecoder
tables.register('decoder_classes', 'ParaformerSANMDecoder')(ParaformerSANMDecoder)
from funasr.models.paraformer.decoder import ParaformerSANMDecoderExport
tables.register('decoder_classes', 'ParaformerSANMDecoderExport')(ParaformerSANMDecoderExport)
from funasr.models.paraformer.decoder import ParaformerSANMDecoderOnlineExport
tables.register('decoder_classes', 'ParaformerSANMDecoderOnlineExport')(ParaformerSANMDecoderOnlineExport)
from funasr.models.paraformer.decoder import ParaformerSANDecoder
tables.register('decoder_classes', 'ParaformerSANDecoder')(ParaformerSANDecoder)
from funasr.models.paraformer.decoder import ParaformerDecoderSANExport
tables.register('decoder_classes', 'ParaformerDecoderSANExport')(ParaformerDecoderSANExport)
from funasr.models.contextual_paraformer.decoder import ContextualParaformerDecoder
tables.register('decoder_classes', 'ContextualParaformerDecoder')(ContextualParaformerDecoder)
from funasr.models.contextual_paraformer.decoder import ContextualParaformerDecoderExport
tables.register('decoder_classes', 'ContextualParaformerDecoderExport')(ContextualParaformerDecoderExport)
from funasr.models.contextual_paraformer.model import ContextualParaformer
tables.register('model_classes', 'ContextualParaformer')(ContextualParaformer)
from funasr.models.ct_transformer.model import CTTransformer
tables.register('model_classes', 'CTTransformer')(CTTransformer)
from funasr.models.ct_transformer_streaming.encoder import SANMVadEncoder
tables.register('encoder_classes', 'SANMVadEncoder')(SANMVadEncoder)
from funasr.models.ct_transformer_streaming.encoder import SANMVadEncoderExport
tables.register('encoder_classes', 'SANMVadEncoderExport')(SANMVadEncoderExport)
from funasr.models.ct_transformer_streaming.model import CTTransformerStreaming
tables.register('model_classes', 'CTTransformerStreaming')(CTTransformerStreaming)
from funasr.models.ctc.model import Transformer
tables.register('model_classes', 'CTC')(Transformer)
from funasr.models.e_branchformer.encoder import EBranchformerEncoder
tables.register('encoder_classes', 'EBranchformerEncoder')(EBranchformerEncoder)
from funasr.models.e_branchformer.model import EBranchformer
tables.register('model_classes', 'EBranchformer')(EBranchformer)
from funasr.models.e_paraformer.decoder import ParaformerSANMDecoder
tables.register('decoder_classes', 'ParaformerSANMDecoder')(ParaformerSANMDecoder)
from funasr.models.e_paraformer.decoder import ParaformerSANMDecoderExport
tables.register('decoder_classes', 'ParaformerSANMDecoderExport')(ParaformerSANMDecoderExport)
from funasr.models.e_paraformer.decoder import ParaformerSANMDecoderOnlineExport
tables.register('decoder_classes', 'ParaformerSANMDecoderOnlineExport')(ParaformerSANMDecoderOnlineExport)
from funasr.models.e_paraformer.decoder import ParaformerSANDecoder
tables.register('decoder_classes', 'ParaformerSANDecoder')(ParaformerSANDecoder)
from funasr.models.e_paraformer.decoder import ParaformerDecoderSANExport
tables.register('decoder_classes', 'ParaformerDecoderSANExport')(ParaformerDecoderSANExport)
from funasr.models.e_paraformer.pif_predictor import PifPredictor
tables.register('predictor_classes', 'PifPredictor')(PifPredictor)
from funasr.models.emotion2vec.model import Emotion2vec
tables.register('model_classes', 'Emotion2vec')(Emotion2vec)
from funasr.models.fsmn_kws.encoder import FSMNConvert
tables.register('encoder_classes', 'FSMNConvert')(FSMNConvert)
from funasr.models.fsmn_kws.model import FsmnKWS
tables.register('model_classes', 'FsmnKWS')(FsmnKWS)
from funasr.models.fsmn_kws.model import FsmnKWSConvert
tables.register('model_classes', 'FsmnKWSConvert')(FsmnKWSConvert)
from funasr.models.fsmn_kws_mt.encoder import FSMNMT
tables.register('encoder_classes', 'FSMNMT')(FSMNMT)
from funasr.models.fsmn_kws_mt.encoder import FSMNMTConvert
tables.register('encoder_classes', 'FSMNMTConvert')(FSMNMTConvert)
from funasr.models.fsmn_kws_mt.model import FsmnKWSMT
tables.register('model_classes', 'FsmnKWSMT')(FsmnKWSMT)
from funasr.models.fsmn_kws_mt.model import FsmnKWSMTConvert
tables.register('model_classes', 'FsmnKWSMTConvert')(FsmnKWSMTConvert)
from funasr.models.fsmn_vad_streaming.encoder import FSMN
tables.register('encoder_classes', 'FSMN')(FSMN)
from funasr.models.fsmn_vad_streaming.encoder import FSMNExport
tables.register('encoder_classes', 'FSMNExport')(FSMNExport)
from funasr.models.fsmn_vad_streaming.model import FsmnVADStreaming
tables.register('model_classes', 'FsmnVADStreaming')(FsmnVADStreaming)
from funasr.models.lcbnet.encoder import TransformerTextEncoder
tables.register('encoder_classes', 'TransformerTextEncoder')(TransformerTextEncoder)
from funasr.models.lcbnet.encoder import SelfSrcAttention
tables.register('encoder_classes', 'FusionSANEncoder')(SelfSrcAttention)
from funasr.models.lcbnet.encoder import ConvPredictor
tables.register('encoder_classes', 'ConvBiasPredictor')(ConvPredictor)
from funasr.models.lcbnet.model import LCBNet
tables.register('model_classes', 'LCBNet')(LCBNet)
from funasr.models.llm_asr.adaptor import Linear
tables.register('adaptor_classes', 'Linear')(Linear)
from funasr.models.llm_asr.adaptor import EncoderProjectorQFormer
tables.register('adaptor_classes', 'QFormer')(EncoderProjectorQFormer)
from funasr.models.llm_asr.adaptor import Transformer
tables.register('adaptor_classes', 'Transformer')(Transformer)
from funasr.models.llm_asr.model import LLMASR
tables.register('model_classes', 'LLMASR')(LLMASR)
from funasr.models.llm_asr.model import LLMASR2
tables.register('model_classes', 'LLMASR2')(LLMASR2)
from funasr.models.llm_asr.model import LLMASR3
tables.register('model_classes', 'LLMASR3')(LLMASR3)
from funasr.models.llm_asr.model import LLMASR4
tables.register('model_classes', 'LLMASR4')(LLMASR4)
from funasr.models.llm_asr_nar.adaptor import Linear
tables.register('adaptor_classes', 'Linear')(Linear)
from funasr.models.llm_asr_nar.model import LLMASRNAR
tables.register('model_classes', 'LLMASRNAR')(LLMASRNAR)
from funasr.models.llm_asr_nar.model import LLMASRNARPrompt
tables.register('model_classes', 'LLMASRNARPrompt')(LLMASRNARPrompt)
from funasr.models.monotonic_aligner.model import MonotonicAligner
tables.register('model_classes', 'MonotonicAligner')(MonotonicAligner)
from funasr.models.normalize.global_mvn import GlobalMVN
tables.register('normalize_classes', 'GlobalMVN')(GlobalMVN)
from funasr.models.normalize.utterance_mvn import UtteranceMVN
tables.register('normalize_classes', 'UtteranceMVN')(UtteranceMVN)
from funasr.models.paraformer_streaming.model import ParaformerStreaming
tables.register('model_classes', 'ParaformerStreaming')(ParaformerStreaming)
from funasr.models.qwen_audio.audio import QwenAudioEncoder
tables.register('encoder_classes', 'QwenAudioEncoder')(QwenAudioEncoder)
from funasr.models.rwkv_bat.rwkv_encoder import RWKVEncoder
tables.register('encoder_classes', 'RWKVEncoder')(RWKVEncoder)
from funasr.models.sa_asr.transformer_decoder import TransformerDecoder
tables.register('decoder_classes', 'TransformerDecoder')(TransformerDecoder)
from funasr.models.sa_asr.transformer_decoder import ParaformerDecoderSAN
tables.register('decoder_classes', 'ParaformerDecoderSAN')(ParaformerDecoderSAN)
from funasr.models.sa_asr.transformer_decoder import LightweightConvolutionTransformerDecoder
tables.register('decoder_classes', 'LightweightConvolutionTransformerDecoder')(LightweightConvolutionTransformerDecoder)
from funasr.models.sa_asr.transformer_decoder import LightweightConvolution2DTransformerDecoder
tables.register('decoder_classes', 'LightweightConvolution2DTransformerDecoder')(LightweightConvolution2DTransformerDecoder)
from funasr.models.sa_asr.transformer_decoder import DynamicConvolutionTransformerDecoder
tables.register('decoder_classes', 'DynamicConvolutionTransformerDecoder')(DynamicConvolutionTransformerDecoder)
from funasr.models.sa_asr.transformer_decoder import DynamicConvolution2DTransformerDecoder
tables.register('decoder_classes', 'DynamicConvolution2DTransformerDecoder')(DynamicConvolution2DTransformerDecoder)
from funasr.models.sanm.decoder import FsmnDecoder
tables.register('decoder_classes', 'FsmnDecoder')(FsmnDecoder)
from funasr.models.sanm.encoder import SANMEncoder
tables.register('encoder_classes', 'SANMEncoder')(SANMEncoder)
from funasr.models.sanm.encoder import SANMEncoderExport
tables.register('encoder_classes', 'SANMEncoderExport')(SANMEncoderExport)
from funasr.models.sanm.encoder import SANMEncoderExport
tables.register('encoder_classes', 'SANMEncoderChunkOptExport')(SANMEncoderExport)
from funasr.models.sanm.model import SANM
tables.register('model_classes', 'SANM')(SANM)
from funasr.models.sanm_kws.model import SanmKWS
tables.register('model_classes', 'SanmKWS')(SanmKWS)
from funasr.models.sanm_kws_streaming.model import SanmKWSStreaming
tables.register('model_classes', 'SanmKWSStreaming')(SanmKWSStreaming)
from funasr.models.scama.decoder import FsmnDecoderSCAMAOpt
tables.register('decoder_classes', 'FsmnDecoderSCAMAOpt')(FsmnDecoderSCAMAOpt)
from funasr.models.scama.encoder import SANMEncoderChunkOpt
tables.register('encoder_classes', 'SANMEncoderChunkOpt')(SANMEncoderChunkOpt)
from funasr.models.scama.model import SCAMA
tables.register('model_classes', 'SCAMA')(SCAMA)
from funasr.models.seaco_paraformer.model import SeacoParaformer
tables.register('model_classes', 'SeacoParaformer')(SeacoParaformer)
from funasr.models.sense_voice.model import SenseVoiceEncoderSmall
tables.register('encoder_classes', 'SenseVoiceEncoderSmall')(SenseVoiceEncoderSmall)
from funasr.models.sense_voice.model import SenseVoiceSmall
tables.register('model_classes', 'SenseVoiceSmall')(SenseVoiceSmall)
from funasr.models.specaug.specaug import SpecAug
tables.register('specaug_classes', 'SpecAug')(SpecAug)
from funasr.models.specaug.specaug import SpecAugLFR
tables.register('specaug_classes', 'SpecAugLFR')(SpecAugLFR)
from funasr.models.transducer.rnn_decoder import RNNDecoder
tables.register('decoder_classes', 'rnn_decoder')(RNNDecoder)
from funasr.models.transducer.rnnt_decoder import RNNTDecoder
tables.register('decoder_classes', 'rnnt_decoder')(RNNTDecoder)
from funasr.models.transformer.encoder import TransformerEncoder
tables.register('encoder_classes', 'TransformerEncoder')(TransformerEncoder)
from funasr.models.uniasr.model import UniASR
tables.register('model_classes', 'UniASR')(UniASR)
from funasr.models.whisper_lid.lid_predictor import LidPredictor
tables.register('lid_predictor_classes', 'LidPredictor')(LidPredictor)
from funasr.models.whisper_lid.model import OpenAIWhisperModel
tables.register('model_classes', 'OpenAIWhisperModel')(OpenAIWhisperModel)
from funasr.models.whisper_lid.model import OpenAIWhisperLIDModel
tables.register('model_classes', 'OpenAIWhisperLIDModel')(OpenAIWhisperLIDModel)
from funasr.tokenizer.char_tokenizer import CharTokenizer
tables.register('tokenizer_classes', 'CharTokenizer')(CharTokenizer)
from funasr.tokenizer.sentencepiece_tokenizer import SentencepiecesTokenizer
tables.register('tokenizer_classes', 'SentencepiecesTokenizer')(SentencepiecesTokenizer)
from funasr.tokenizer.hf_tokenizer import HuggingfaceTokenizer
tables.register('tokenizer_classes', 'HuggingfaceTokenizer')(HuggingfaceTokenizer)
from funasr.tokenizer.whisper_tokenizer import WhisperTokenizer
tables.register('tokenizer_classes', 'WhisperTokenizer')(WhisperTokenizer)
from funasr.tokenizer.whisper_tokenizer import SenseVoiceTokenizer
tables.register('tokenizer_classes', 'SenseVoiceTokenizer')(SenseVoiceTokenizer)
