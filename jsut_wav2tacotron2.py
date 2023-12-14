import os
import librosa
import soundfile as sf

# パス
in_paths = [
    'jsut_ver1.1/basic5000/',
    'jsut_ver1.1/countersuffix26/',
    'jsut_ver1.1/loanword128/',
    'jsut_ver1.1/onomatopee300/',
    'jsut_ver1.1/precedent130/',
    'jsut_ver1.1/repeat500/',
    'jsut_ver1.1/travel1000/',
    'jsut_ver1.1/utparaphrase512/',
    'jsut_ver1.1/voiceactress100/']
out_path = 'wav/'

# 出力フォルダの準備
os.makedirs(out_path, exist_ok=True)

# wavの変換の関数
def convert(in_path):
    filenames = os.listdir(in_path+'wav/')
    for filename in filenames:
        print(in_path+'wav/'+filename)
        y, sr = librosa.core.load(in_path+'wav/'+filename, sr=22050, mono=True)
        sf.write(out_path+'wav/'+filename, y, sr, subtype="PCM_16")

# wavの変換
for in_path in in_paths:
    convert(in_path)
