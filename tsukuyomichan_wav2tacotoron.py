import os
import librosa
import soundfile as sf

# パス
in_path = "tsukuyomichan_copas/02_WAV/"
out_path = "wav_tsukuyomichan/"

# 出力フォルダの準備
os.makedirs(out_path, exist_ok=True)

# wavの変換の関数
filenames = os.listdir(in_path)
for filename in filenames:
    print(in_path+filename)
    y, sr = librosa.core.load(in_path+filename, sr=22050, mono=True)
    sf.write(out_path+filename, y, sr, subtype="PCM_16")