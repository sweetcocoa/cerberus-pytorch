import glob
import os
from collections import defaultdict

import torch
import torch.nn as nn
import torchaudio
import numpy as np
from omegaconf import OmegaConf
from dataset.midi import parse_midi, parsed_midi_to_roll

torchaudio.set_audio_backend("sox_io")

class Slakh2100(torch.utils.data.Dataset):
    def __init__(self, 
                 data_dir, 
                 phase, 
                 inst=['Piano', 'Bass', 'Drums'], 
                 sr=44100, 
                 duration=2.0, 
                 hop_length=256,
                 num_pitches=88,
                 use_cache=True,
                 random=True,
                 n_fft=1024,
                 midi_min=21,
                 ):
        """
        data_dir : "/mnt/ssd3/mlproject/data/slakh2100_flac/"
        phase    : "train"
        """
        super().__init__()
        self.tracks = sorted(glob.glob(os.path.join(data_dir, phase) + "/*"))
        self.sr = sr
        self.inst = inst
        self.phase = phase
        self.data_dir = data_dir
        self.duration = duration
        self.hop_length = hop_length
        self.random = random
        self.num_pitches = num_pitches
        self.meta = None
        self.midi_min = midi_min

        x = torch.zeros(int(self.sr * self.duration))
        X = torch.stft(x, n_fft, hop_length)
        self.num_tr_frames = X.shape[1]

        cache_location = os.path.join(data_dir, f"meta_{phase}_{inst}_v1.pth")
        if os.path.exists(cache_location) and use_cache:
            self.meta = torch.load(cache_location)
        else:
            print(f"Creating slakh2100 Cache for {inst}..")
            self.meta = self.get_meta(self.tracks)
            torch.save(self.meta, cache_location)
            print("Done")
        
        self.labels = [None for track in range(len(self.tracks))]

    def get_label(self, idx):
        """
        onset / offset / frame piano roll of the midi file, 
        See : parse_midi, parsed_miti_to_roll
        """
        if self.labels[idx] is None:
            track = self.tracks[idx]
            info = self.meta[idx]['info']
            inst_track = self.meta[idx]['inst_track']
            track_labels = dict()  # track_labels['Piano'] = np.array(n_steps, pitch)

            for inst in self.inst:
                if inst in inst_track:
                    stem = inst_track[inst]
                    midi_path = track + f"/MIDI/{stem}.mid"
                    midi_cache = midi_path.replace(".mid", ".pt")
                    if os.path.exists(midi_cache):
                        label = torch.load(midi_cache)
                    else:
                        mid_np = parse_midi(midi_path)
                        label = parsed_midi_to_roll(mid_np, info.num_frames, hop_length=self.hop_length, sample_rate=self.sr, num_pitches=self.num_pitches, midi_min=self.midi_min, hops_in_offset=2, hops_in_onset=2)
                        torch.save(label, midi_cache)
                    track_labels[inst] = label
            self.labels[idx] = track_labels
            return track_labels
        else:
            return self.labels[idx]

    def get_meta(self, tracks):
        """
        cfg['info'] = torchaudio.info (cfg['info'].num_frames = duration * sample_rate )
        cfg['inst_track']
        cfg['inst_track']['Piano'] = "S01"
        cfg['inst_track']['Bass'] = "S02"
        cfg['inst_track']['Drums'] = "S04"
        """

        cfgs = []
        for k, track in enumerate(tracks):
            info = torchaudio.info(track + "/mix.flac")
            cfg = dict(info=info)
            
            track_cfg = OmegaConf.load(track + "/metadata.yaml")
            inst_track = dict()
            for i in track_cfg.stems:
                if track_cfg.stems[i].audio_rendered and track_cfg.stems[i].inst_class in self.inst and track_cfg.stems[i].midi_saved:
                    if track_cfg.stems[i].inst_class in inst_track:
                        prev_i = inst_track[track_cfg.stems[i].inst_class]
                        if track_cfg.stems[prev_i].program_num > track_cfg.stems[i].program_num:
                            inst_track[track_cfg.stems[i].inst_class] = i    
                    else:
                        inst_track[track_cfg.stems[i].inst_class] = i
            cfg['inst_track'] = inst_track
            cfgs.append(cfg)
        return cfgs

    def __getitem__(self, idx):
        track = self.tracks[idx]
        info = self.meta[idx]['info']
        inst_track = self.meta[idx]['inst_track']

        if self.random:
            step_begin = np.random.randint(info.num_frames - int(self.duration*self.sr) - 1) // self.hop_length
        else:
            frame_offset = (info.num_frames - int(self.duration*self.sr) - 1)//3
            step_begin = frame_offset // self.hop_length
            
        n_steps = self.num_tr_frames
        step_end = step_begin + n_steps

        raw_begin = step_begin * self.hop_length

        ys = []
        transcripts = []
        onsets = []
        offsets = []

        for inst in self.inst:
            if inst in inst_track:
                stem = inst_track[inst]
                y, sr = torchaudio.load(track + f"/stems/{stem}.flac", frame_offset=raw_begin, num_frames=int(self.duration*self.sr))
                assert sr == self.sr

                if len(y[0]) < int(self.sr * self.duration):
                    y = nn.functional.pad(y, (0, int(self.sr * self.duration) - len(y[0])))
                
                label = self.get_label(idx)[inst] # (steps, pitch)
                midi_piece = label['label'][step_begin:step_end, :]
               
                ys.append(y)
                transcripts.append((midi_piece > 1).float().T)
                onsets.append((midi_piece == 3).float().T)
                offsets.append((midi_piece == 1).float().T)
            else:
                # 0 tensor if no source is available on that track
                ys.append(torch.zeros((1,  int(self.sr * self.duration))))
                roll = np.zeros((self.num_pitches , self.num_tr_frames), dtype=np.float32)
                transcripts.append(torch.Tensor(roll))
                onsets.append(torch.Tensor(roll))
                offsets.append(torch.Tensor(roll))          
                
        separation_gt = torch.cat(ys)
        transcripts = torch.stack(transcripts)
        onsets = torch.stack(onsets)
        offsets = torch.stack(offsets)
        mix = torch.sum(separation_gt, dim=0)

        sample = dict(
            separation_gt=separation_gt,
            mix=mix,
            transcripts_gt=transcripts,
            onsets_gt=onsets,
            offsets_gt=offsets
        )

        return sample
    
    def __len__(self):
        return len(self.tracks)