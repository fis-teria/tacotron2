import os
import librosa
import soundfile as sf

# サンプリングビットの変換
in_path = 'archive/meian/'
out_path = 'filelists/meian/'
filenames = os.listdir(in_path)
for filename in filenames:
   print(out_path+filename)
   y, sr = librosa.core.load(in_path+filename, sr=22050, mono=True) # 22050Hz、モノラルで読み込み
   sf.write(out_path+filename, y, sr, subtype="PCM_16") #16bitで書き込み
