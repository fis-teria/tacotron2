import matplotlib
import matplotlib.pylab as plt
import IPython.display as ipd
import sys
sys.path.append('waveglow/')
import numpy as np
import torchaudio
import torch
from torch import mps
from hparams_v2 import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from train_v2 import load_model
from text import text_to_sequence
from denoiser_v2 import Denoiser

def plot_data(data, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        # axes[i].imshow(data[i], aspect='auto', origin='bottom', 
        axes[i].imshow(data[i], aspect='auto', origin='lower', 
                       interpolation='none')

hparams = create_hparams()

checkpoint_path = "outdir2/checkpoint_37000"
model = load_model(hparams)
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
# _ = model.cuda().eval().half()
_ = model.to("cpu").eval()
# _ = model.to("mps").eval().half()

waveglow_path = 'waveglow_256channels_universal_v5.pt'
waveglow = torch.load(waveglow_path)['model']
# waveglow.cuda().eval().half()
waveglow.to("cpu").eval()
# waveglow.to("mps").eval().half()
for k in waveglow.convinv:
    k.float()
denoiser = Denoiser(waveglow)

text = "konnitiwa"
sequence = np.array(text_to_sequence(text, ['basic_cleaners']))[None, :]
sequence = torch.autograd.Variable(
#     torch.from_numpy(sequence)).cuda().long()
    torch.from_numpy(sequence)).to("cpu").long()

mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
plot_data((mel_outputs.float().data.cpu().numpy()[0],
           mel_outputs_postnet.float().data.cpu().numpy()[0],
           alignments.float().data.cpu().numpy()[0].T))

with torch.no_grad():
    audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)
# ipd.Audio(audio[0].data.cpu().numpy(), rate=hparams.sampling_rate)

audio_tensor = torch.from_numpy(audio[0].data.cpu().numpy())
audio_tensor = audio_tensor.unsqueeze(0)  # 2次元のテンソルに変換
torchaudio.save(filepath='result.wav', src=audio_tensor, sample_rate=hparams.sampling_rate)
