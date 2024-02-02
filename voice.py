import sys
sys.path.append('waveglow/')
print(sys.path)
import numpy as np
import torch
import torchaudio

import pyopenjtalk

from hparams_v2 import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
from denoiser import Denoiser

import pygame
import time


hparams = create_hparams()
hparams.sampling_rate = 22050

checkpoint_path = "outdir3/checkpoint_5000"
model = load_model(hparams)
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
_ = model.cuda().eval().float()

waveglow_path = 'waveglow_256channels_universal_v5.pt'
waveglow = torch.load(waveglow_path)['model']
waveglow.cuda().eval().float()
for k in waveglow.convinv:
    k.float()
denoiser = Denoiser(waveglow)

text = "あはーん　だいすきなんだなーん　ちゅちゅちゅ　ちゅちゅちゅのちゅーだよ"
phones = pyopenjtalk.g2p(text, kana=False)
phones = phones.replace('pau',',')
phones = phones.replace(' ','')
phones = phones + '.'
print(phones)
sequence = np.array(text_to_sequence(phones, ['basic_cleaners']))[None, :]
#sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
sequence = torch.autograd.Variable(
    torch.from_numpy(sequence)).cuda().long()
print(sequence)

mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
mel_outputs = mel_outputs.to(torch.float32)
mel_outputs_postnet = mel_outputs_postnet.to(torch.float32)
_ = _.to(torch.float32)
alignments = alignments.to(torch.float32)


with torch.no_grad():
    audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)


audio_tensor = torch.from_numpy(audio[0].data.cpu().numpy())
audio_tensor = audio_tensor.unsqueeze(0)  # 2次元のテンソルに変換
audio_tensor = audio_tensor.to(torch.float32)
torchaudio.save(uri='result.wav', src=audio_tensor, sample_rate=hparams.sampling_rate, format='wav')

pygame.mixer.init()
pygame.mixer.music.load('result.mp3')
pygame.mixer.music.play()
time.sleep(5)