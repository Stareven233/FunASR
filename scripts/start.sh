# export LD_LIBRARY_PATH=`python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))'`
conda activate whisper
export LD_LIBRARY_PATH=/home/cxp/data/env/whisper/lib/python3.10/site-packages/nvidia/cublas/lib:/home/cxp/data/env/whisper/lib/python3.10/site-packages/nvidia/cudnn/lib/home/cxp/data/env/whisper/lib/python3.10/site-packages/nvidia/cublas/lib:/home/cxp/data/env/whisper/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
# export HF_ENDPOINT=https://hf-mirror.com
python scripts/main.py --path=../media/konosuba.mp3
screen -S whisper python scripts/main.py --path=../media/rie.mp4
screen -S whisper python scripts/main.py --path=/data/cxp/toys/fish-speech/references/0/Megumin_Main4_1_5_9.wav
