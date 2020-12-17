# python inference.py config_path weight_ckpt input_wav output_dir 

import os
import sys

import pytorch_lightning as pl
import torch
import torchaudio
import matplotlib.pyplot as plt

import librosa
import librosa.display
from omegaconf import OmegaConf

import pretty_midi as pm
import numpy as np

from network.cerberus_wrapper import CerberusWrapper

config_path = sys.argv[1]   # "lightning_logs/experiment_name/version_0/hparams.yaml"
weight_path = sys.argv[2]   # "lightning_logs/experiment_name/version_0/checkpoints/last.ckpt"
input_wav = sys.argv[3]
output_dir = sys.argv[4]

config = OmegaConf.load(config_path)

y, sr = torchaudio.load(input_wav, frame_offset=config.sr*90, num_frames=config.sr*20)
assert sr == config.sr

n = CerberusWrapper.load_from_checkpoint(weight_path)

rt = n.get_transcripts(y)

fs = config.sr / config.hop_length
plt.rcParams.update({"figure.facecolor":  (1.0, 1.0, 1.0, 1.0),
    "axes.facecolor":    (1.0, 1.0, 1.0, 1.0),
})

os.makedirs(output_dir, exist_ok=True)

for i, inst in enumerate(config.inst):
    plt.title(inst + "_Activation")
    librosa.display.specshow(rt[0][i].float().detach().numpy(), hop_length=1, sr=int(fs), x_axis='time', y_axis='cqt_note',
                             fmin=pm.note_number_to_hz(config.midi_min))
    plt.savefig(os.path.join(output_dir, f"transcript_activation_{inst}.jpg"))
    threshold = 0.8
    if inst == "Drums":
        threshold = threshold / 4
    
    plt.title(inst + "_Binary")
    librosa.display.specshow(rt[0][i].float().detach().numpy() > threshold, hop_length=1, sr=int(fs), x_axis='time', y_axis='cqt_note',
                             fmin=pm.note_number_to_hz(config.midi_min))
    plt.savefig(os.path.join(output_dir, f"transcript_threshold_{inst}_{threshold}.jpg"))

pmidi = n.multiple_piano_roll_to_pretty_midi(rt[0][:3])
pmidi.write(os.path.join(output_dir, "transcripted_midi.mid"))

source_hat = n.get_separated_sources(y)

for i, inst in enumerate(config.inst):
    audio_save_path = os.path.join(output_dir, f"separated_{inst}.wav")
    torchaudio.save(audio_save_path, source_hat[0, i].detach().unsqueeze(0), sample_rate=config.sr)
