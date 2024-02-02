import matplotlib
#%matplotlib inline
import matplotlib.pyplot as plt

import wave
import pygame

import IPython.display as ipd

import sys
sys.path.append('waveglow/')
import numpy as np
import torch
import torchaudio

import pyopenjtalk
import time

from hparams_v2 import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
from denoiser import Denoiser

def play_mp3(filename):
    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()



hparams = create_hparams()
hparams.sampling_rate = 22050

#checkpoint_path = "tacotron2_statedict.pt"
checkpoint_path = "outdir3/checkpoint_5000"
model = load_model(hparams)
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
_ = model.cuda().eval().half()

waveglow_path = 'waveglow_256channels_universal_v5.pt'
waveglow = torch.load(waveglow_path)['model']
#print(waveglow)
#for k in dir(waveglow):
#    print(k)
waveglow.cuda().eval().float() #floatにしないとデータのビット数が足らない
for k in waveglow.convinv:
    k.float()
denoiser = Denoiser(waveglow)

#text = "Waveglow is really awesome!"
text = "まだ助かる、まだ助かる、そーれ！ここマダガスカル！"
#print("入力してね>>>")
#text = input()
phones = pyopenjtalk.g2p(text, kana=False)
phones = phones.replace('pau',',')
phones = phones.replace(' ','')
phones = phones + '.'
#print(phones)
sequence = np.array(text_to_sequence(phones, ['basic_cleaners']))[None, :]
#sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
sequence = torch.autograd.Variable(
    torch.from_numpy(sequence)).cuda().long()
#print(sequence)

mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
#plot_data((mel_outputs.float().data.cpu().numpy()[0],
#           mel_outputs_postnet.float().data.cpu().numpy()[0],
#           alignments.float().data.cpu().numpy()[0].T))

mel_outputs = mel_outputs.to(torch.float32)
mel_outputs_postnet = mel_outputs_postnet.to(torch.float32)
_ = _.to(torch.float32)
alignments = alignments.to(torch.float32)
#print(mel_outputs_postnet.type())
#data = [mel_outputs.float().data.cpu().numpy()[0], mel_outputs_postnet.float().data.cpu().numpy()[0], alignments.float().data.cpu().numpy()[0].T]
#print('data=>')
#print(data)
#print('data_elements=>')
#print(mel_outputs.float().data.cpu().numpy()[0])
#print(mel_outputs_postnet.float().data.cpu().numpy()[0])
#print(alignments.float().data.cpu().numpy()[0].T)

#figsize=(16, 4)
#fig, axes = plt.subplots(1, len(data), figsize=figsize)
#for i in range(len(data)):
#    axes[i].imshow(data[i], aspect='auto', origin='lower', interpolation='none')
#fig.savefig('graph.png')
#plt.show()

with torch.no_grad():
    #print(mel_outputs_postnet)
    audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)

#print(audio[0].data.cpu().numpy())
#ipd.Audio(audio[0].data.cpu().numpy(), rate=hparams.sampling_rate)

audio_tensor = torch.from_numpy(audio[0].data.cpu().numpy())
audio_tensor = audio_tensor.unsqueeze(0)  # 2次元のテンソルに変換
audio_tensor = audio_tensor.to(torch.float32)
torchaudio.save(uri='result.mp3', src=audio_tensor, sample_rate=hparams.sampling_rate, format='mp3')

play_mp3('result.mp3')
delay(5)