## 查看支持的硬件加速方法
ffmpeg -hwaccels
ffmpeg -codecs | findstr "h264"
<!-- 查看所支持的 h264 编解码器
 DEV.LS h264 ... (decoders: h264 h264_qsv h264_cuvid)  (encoders: libx264 libx264rgb h264_amf h264_mf h264_nvenc h264_qsv h264_vaapi h264_vulkan) -->

## 查看视频信息
查看视频文件的基本信息
ffprobe -i input.mp4

### 查看视频文件的格式信息
ffprobe -i input.mp4 -show_format
ffprobe -i output.mp4 -show_format

### 查看视频文件的码流信息
ffprobe -i input.mp4 -show_streams

### 查看视频文件的帧信息
ffprobe -i input.mp4 -show_frames

## 合并音视频
要使用FFmpeg合并音频和视频，首先确保要合并的视频文件中不包含音频。然后，可以使用以下命令：

ffmpeg -i video.mp4 -i audio.aac -c:v copy -c:a copy output.mp4
这里的命令参数解释如下：

-i video.mp4 指定视频输入文件。

-i audio.aac 指定音频输入文件。

-c:v copy 表示复制视频流，不进行编码。

-c:a copy 表示复制音频流，不进行编码。

output.mp4 是合并后生成的文件名。

---
$n='z0#GTJ-029 縄処女崩壊 篠宮ゆり'
ffmpeg -i "$n.mp4" -i "$n.aac" -c:v copy -c:a copy "${n}_merged.mp4"

$n='环卫工提意见，领导咆哮：“你Z反呐？罚500、1000、开除！”'
ffmpeg -i "$n.mp4" -i "$n.m4a" -c:v copy -c:a copy "${n}_merged.mp4"

## 提取音频
ffmpeg -i sample.mp4 -q:a 0 -map a sample.mp3
其中，-i 指定输入文件，-q:a 0 表示最高质量，-map a 表示只提取音频流

不想重新编码音频，可以使用以下命令：
ffmpeg -i input-video.avi -vn -acodec copy output-audio.aac
-vn 表示不包含视频，-acodec copy 表示直接复制音频流
---
$n='MIAD-924 俺の妹とお前の妹どっちがエロいか交換して中出ししまくってみないか？'
ffmpeg -i "$n.mp4" -vn -c:a copy "${n}.aac"
ffmpeg -i "$n.mp4" -vn -c:a copy -ss 00:00:60 -to 00:02:00 "${n}.aac"
ffmpeg -i "$n.mp4" -vn -c:a aac "${n}.aac"
ffmpeg -i "$n.mp4" -q:a 0 -map a "${n}.aac"

## 提取视频
$n='z0#GTJ-029 縄処女崩壊 篠宮ゆり'
ffmpeg -i "$n.mp4" -vcodec copy -an "{$n}_v.mp4"

## 字幕格式转换
使用ffmpeg可以很轻松的把ass/vtt/lyric转换为srt文件，命令如下：

ffmpeg -i a.ass b.srt
ffmpeg -i c.vtt d.srt
ffmpeg -i e.lyric f.srt

## 视频格式转换
ffmpeg -i input.mkv output.mp4

ffmpeg -i input.mkv -map 0 -c copy output.mp4
ffmpeg -i input.mkv -map 0 -c copy -movflags +faststart output.mp4
<!-- 
-c copy: 直接复制原始的音视频流（不重新编码），速度最快且质量无损。需要原 MKV 中的编码格式（如 H.264/H.265 视频、AAC 音频等）兼容 MP4 标准。
-movflags +faststart: 将 MP4 的元数据移动到文件开头，方便网络流媒体播放（如网页在线播放）。转换速度会略微变慢，但远快于重新编码。无需在线播放可省略此参数。 
-map 0: 确保复制所有流（默认可能只选一条音轨和字幕）
-->

ffmpeg -hwaccel cuvid -i input.mkv -c:v h264_nvenc output.mp4
ffmpeg -i input.mkv -vf scale=1920:1080 -r 30 -c:v h264_nvenc -b:v 5000k -preset medium -cq 0 -c:a copy output.mp4
<!-- 
-hwaccel cuvid: 指定使用 NVIDIA 的 CUVID 硬件加速进行视频解码。
-c:v h264_nvenc: 指定使用 NVIDIA 的 NVENC 编码器进行视频编码，将视频编码为 H.264 格式。
-cq: 调整视频质量，取值范围为 0 - 31，类似于 -crf 参数，值越小质量越高，23 是一个常用的默认值。
-vf scale=1920:1080: 使用 NVIDIA Performance Primitives（NPP）进行视频缩放，将视频分辨率调整为 1920x1080。scale=2048:-1，宽度为2048，保持长宽比
-r 30: 设置输出视频的帧率为 30fps。
-preset fast: 设置编码预设为 fast，在保证一定质量的前提下提高编码速度。你还可以根据需要选择其他预设，如 slow 以获得更高质量但较慢的编码速度。
-rc vbr_hq: 设置码率控制模式为可变比特率高质量模式（VBR HQ），这种模式可以在不同场景下动态调整比特率，以达到较好的质量和文件大小平衡。
-b:v 5000k: 设置音频的比特率为 5000kbps
-b:a 128k: 设置音频的比特率为 128kbps。 -->

ffmpeg -i input.mkv -vf scale=1920:1080 -r 30 -c:v h264_nvenc -q:v 0 -b:v 4000k -c:a copy output.mp4
<!-- -q:v：表示存储jpeg的图像质量。似乎设置为0后面的-b:v才会生效 -->
