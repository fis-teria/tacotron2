import os

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
out_path = 'filelists/'

# 出力フォルダの準備
os.makedirs(out_path, exist_ok=True)

# transcriptの変換
with open(out_path+'transcript_utf8.txt', 'w') as wf:
    for in_path in in_paths:
        with open(in_path+'transcript_utf8.txt', 'r') as rf:
            wf.write(rf.read())
