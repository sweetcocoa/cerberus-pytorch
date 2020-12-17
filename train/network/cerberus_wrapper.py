import sys
import os
from collections import defaultdict

import torch
import torch.nn as nn 
import torch.optim as optim
import pytorch_lightning as pl
import torchaudio
from omegaconf import OmegaConf
import librosa.display
import pretty_midi as pm
import matplotlib.pyplot as plt 

from network.cerberus import Cerberus
from network.transform_layer import ISTFT

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")

from loss.mask_inference_loss import MaskInferenceLoss
from loss.deep_cluster_loss import DeepClusterLoss
from metrics.transcript_metric import TrMetrics
from utils.dsp import realimag, apply_masks
from dataset.slakh2100 import Slakh2100
from dataset.midi import piano_roll_to_pretty_midi, program_dict

class CerberusWrapper(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters(self.config)
        
        self.stft = torchaudio.transforms.Spectrogram(n_fft=config.n_fft, hop_length=config.hop_length, power=None)
        self.istft = ISTFT(config)

        if 'sep' in self.config.heads:
            self.mask_inference_loss = MaskInferenceLoss(config.num_inst)
            self.mask_metric = None  # Not implemented.
        else:
            self.mask_inference_loss = None
            self.mask_metric = None

        if 'dc' in self.config.heads:
            self.deep_clustering_loss = DeepClusterLoss()
            self.clustering_metric = None # Not implemented.
        else:
            self.deep_clustering_loss = None
            self.clustering_metric = None

        if 'tr' in self.config.heads:
            self.transcription_loss = nn.BCELoss()
            self.transcription_metric = TrMetrics(config)
        else: 
            self.transcription_loss = None
            self.transcription_metric = None

        self.cerberus = Cerberus(config)

        # auto lr find
        self.lr = config.lr

        self.saved_gt_to_tensorboard = False

    def val_dataloader(self):
        config = self.config

        valid_ds = Slakh2100(data_dir=config.data_dir, 
                             phase="validation", 
                             inst=config.inst, 
                             duration=config.duration, 
                             use_cache=True, 
                             sr=config.sr, 
                             random=False,
                             n_fft=config.n_fft, 
                             hop_length=config.hop_length,
                             midi_min=config.midi_min)
        valid_dl = torch.utils.data.DataLoader(valid_ds, shuffle=False, batch_size=config.batch_size, num_workers=config.num_workers, pin_memory=True)
        return valid_dl

    def train_dataloader(self):
        config = self.config

        train_ds = Slakh2100(data_dir=config.data_dir, 
                             phase="train", 
                             inst=config.inst, 
                             duration=config.duration, 
                             use_cache=True, 
                             sr=config.sr, 
                             random=True,
                             n_fft=config.n_fft, 
                             hop_length=config.hop_length,
                             midi_min=config.midi_min)

        train_dl = torch.utils.data.DataLoader(train_ds, shuffle=True, batch_size=config.batch_size, num_workers=config.num_workers, pin_memory=True)
        return train_dl

    def test_dataloader(self):
        config = self.config

        test_ds = Slakh2100(data_dir=config.data_dir, 
                             phase="test", 
                             inst=config.inst, 
                             duration=4., 
                             use_cache=True, 
                             sr=config.sr, 
                             random=False,
                             n_fft=config.n_fft, 
                             hop_length=config.hop_length,
                             midi_min=config.midi_min)
        test_dl = torch.utils.data.DataLoader(test_ds, shuffle=False, batch_size=2, num_workers=config.num_workers, pin_memory=True)
        return test_dl


    def batch_preprocessing(self, mini_batch:dict):
        """
        mix -> its stft, magnitude, phase
        separation(source) -> stft, magnitude

        mini_batch.keys()
        >> "mix", "separation_gt", "transcript_gt"
        mini_batch = self.batch_preprocessing(mini_batch)
        mini_batch.keys()
        >> "mix", "separation_gt", "transcript_gt", 
        "mix_mag", "mix_phase", "mix_stft",
        "separation_gt_stft", "separation_gt_mag"
        """

        mix_stft = self.stft(mini_batch['mix'])  # (batch, freq, time, 2)
        mix_mag, mix_phase = torchaudio.functional.magphase(mix_stft, power=1.0)

        mix_mag = mix_mag.permute(0, 2, 1)  # (batch, time, freq)

        sep_stft = self.stft(mini_batch['separation_gt'])
        sep_mag, sep_phase = torchaudio.functional.magphase(sep_stft, power=1.0)
        sep_mag = sep_mag.permute(0, 1, 3, 2)
        
        mini_batch['mix_stft'] = mix_stft
        mini_batch['mix_mag'] = mix_mag
        mini_batch['mix_phase'] = mix_phase
        mini_batch['separation_gt_stft'] = sep_stft
        mini_batch['separation_gt_mag'] = sep_mag

        return mini_batch

    def sample_preprocessing(self, mix:torch.Tensor):
        prep = dict()

        # mix : (batch, time)
        mix_stft = self.stft(mix)
        
        # stft : (batch, freq, time, 2)
        mix_mag, phase = torchaudio.functional.magphase(mix_stft, power=1.0)

        # mag : (batch, freq, time)

        mix_mag = mix_mag.permute(0, 2, 1) # mag : (batch, time, freq)
        
        prep['mix_mag'] = mix_mag
        prep['phase'] = phase
        return prep

    def get_separated_sources(self, mix:torch.Tensor):
        """
        Just for inference
        audio(batch, time) 입력으로 받아서 num_inst만큼 return (batch, inst, time)
        """
        prep = self.sample_preprocessing(mix)
        mix_mag, phase = prep['mix_mag'], prep['phase']
        rt = self(mix_mag)

        mask = rt['separation_mask']
        sep_mag_hat = apply_masks(mask, mix_mag, self.config.num_inst) #(batch, inst, time, freq)

        # (batch, inst, time, freq) -> (batch, inst, freq, time)
        sep_mag_hat = sep_mag_hat.permute(0, 1, 3, 2)

        # use mix phase to do istft
        b, f, t = phase.shape
        phase_repeat = phase.repeat_interleave(self.config.num_inst, 0).view(b, self.config.num_inst, f, t) # (batch, inst, freq, time)

        # (batch, inst, freq, time, 2)
        sep_stft_hat = realimag(sep_mag_hat, phase_repeat)

        # (batch, inst, time)
        separated_wavs = self.istft(sep_stft_hat)
        return separated_wavs

    def get_transcripts(self, mix):
        """
        Just for inference
        
        mix : audio(batch, time) 
        
        return : transcription (batch, inst, pitch, time)
        """
        prep = self.sample_preprocessing(mix)
        mix_mag = prep['mix_mag']
        rt = self(mix_mag)
        transcripts = rt['transcripts']

        # (batch, inst, pitch, time)
        return transcripts

    def common_step(self, mini_batch:dict, phase:str):
        mini_batch = self.batch_preprocessing(mini_batch)

        rt = self(mini_batch['mix_mag'])
        
        total_loss = 0.
        log = dict()

        if self.mask_inference_loss is not None:
            mask_inference_loss = self.mask_inference_loss(rt['separation_mask'], mini_batch['mix_mag'], mini_batch['separation_gt_mag'])
            total_loss += self.config.loss_beta * mask_inference_loss
            log[f'{phase}_mask_inference_loss'] = mask_inference_loss

        if self.deep_clustering_loss is not None:
            deep_clustering_loss = self.deep_clustering_loss(rt['embedding'], mini_batch['separation_gt_mag'])
            total_loss += self.config.loss_alpha * deep_clustering_loss
            log[f'{phase}_dc_loss'] = deep_clustering_loss

        if self.transcription_loss is not None:
            transcription_loss = self.transcription_loss(rt['transcripts'], mini_batch['transcripts_gt'])
            total_loss += self.config.loss_gamma * transcription_loss
            log[f'{phase}_tr_loss'] = transcription_loss

        log[f'{phase}_total_loss'] = total_loss
        self.log_dict(log, on_epoch=True, on_step=False)
        return total_loss

    def forward(self, mix_mag):
        rt = self.cerberus(mix_mag)
        return rt

    def on_train_start(self):
        metrics = {k: v for d in OmegaConf.to_container(self.config.metrics) for k, v in d.items()}
        if not isinstance(self.logger, pl.loggers.base.DummyLogger):
            # dummy logger일 때(auto lr find) 아래에서 에러나서
            self.logger.log_hyperparams(self.hparams, metrics=metrics)

    def training_step(self, mini_batch, batch_idx):
        phase = "train"
        total_loss = self.common_step(mini_batch, phase)
        return total_loss

    def validation_step(self, mini_batch, batch_idx):
        phase = "valid"
        total_loss = self.common_step(mini_batch, phase)
        return total_loss

    def test_step(self, mini_batch, batch_idx):
        """
        batch -> transcript, separation -> transcription metric
        Warning : TOO SLOW
        """

        phase = "test"
        batch_size = mini_batch['mix'].shape[0]
        transcript_est = self.get_transcripts(mini_batch['mix'])

        target_metrics = ['Precision', 'Recall', 'Accuracy']

        # metrics['piano'] = {'Precision' : [0.33, 0.324], 'Recall' : [0.24, 0.15], 'Accuracy' : [0.33, 0.22]}
        metrics = dict()
        for inst in self.config.inst:
            # inst : str
            metrics[inst] = defaultdict(list)

        for i, inst in enumerate(self.config.inst):
            # inst : int
            for b in range(batch_size):        
                tr = transcript_est[b, i]
                gt = mini_batch['transcripts_gt'][b, i]
                threshold = self.config.transcription_threshold / 4 if inst == "Drums" else self.config.transcription_threshold
                result = self.transcription_metric(tr, gt, threshold)

                for tm in target_metrics:
                    metrics[inst][tm].append(result[tm])
        
        return metrics

    def test_epoch_end(self, test_out):
        target_metrics = ['Precision', 'Recall', 'Accuracy']
        num_samples = 0
        for metric in test_out:
            num_samples += len(metric[self.config.inst[0]][target_metrics[0]])

        # final_metrics['piano'] = {'precision': 0.324, 'Recall': .2343 ... }        
        final_metrics = dict()
        for inst in self.config.inst:
            # inst : str
            final_metrics[inst] = defaultdict(float)

        for out in test_out:
            for inst in self.config.inst:
                for tm in target_metrics:
                    final_metrics[inst][tm] += sum(out[inst][tm])
            
        for inst in self.config.inst:
            for tm in target_metrics:
                final_metrics[inst][tm] /= num_samples

        self.log_dict(final_metrics)

    def multiple_piano_roll_to_pretty_midi(self, rt):
        # rt = (inst, freq, time) (float tensor)
        # return : prettymidi object

        config = self.config
        pmidi = pm.PrettyMIDI()

        for i in range(config.num_inst):
            threshold = config.transcription_threshold / 4 if config.inst[i] == "Drums" else config.transcription_threshold
            is_drum = config.inst[i] == "Drums"

            pr = rt[i].cpu().float().detach() > threshold
            pr_pad = torch.nn.functional.pad(pr, (0, 0, config.midi_min, 128-config.num_pitches-config.midi_min))
            instrument = piano_roll_to_pretty_midi(pr_pad.int().numpy(), fs=config.sr/config.hop_length, program=program_dict[config.inst[i]]).instruments[0]
            instrument.is_drum = is_drum
            pmidi.instruments.append(instrument)
        
        return pmidi
        
    def validation_epoch_end(self, val_out: list):
        """
        Save Audio / Transcription results on every validation
        """
        writer = self.logger.experiment
        
        mini_batch = next(iter(self.trainer.val_dataloaders[0]))
        for k,v in mini_batch.items():
            mini_batch[k] = v.cuda()

        sample_audio, sr = torchaudio.load(self.config.sample_audio.path, frame_offset=self.config.sample_audio.offset, num_frames=self.config.sample_audio.num_frames)        
        sample_audio = sample_audio.cuda()

        if self.cerberus.separation_head is not None:
            y_hat = self.get_separated_sources(mini_batch['mix'])
            for i in range(self.config.num_inst):
                writer.add_audio(f"{self.config.inst[i]}_hat", y_hat[4, i].cpu(), self.trainer.global_step , sample_rate=self.config.sr)
                if not self.saved_gt_to_tensorboard:
                    writer.add_audio(f"{self.config.inst[i]}_gt", mini_batch['separation_gt'][4, i].cpu(), self.trainer.global_step , sample_rate=self.config.sr)

            y_hat = self.get_separated_sources(sample_audio)
            if not self.saved_gt_to_tensorboard:
                writer.add_audio(f"(real)gt", sample_audio[0].cpu(), self.trainer.global_step , sample_rate=self.config.sr)

            for i in range(self.config.num_inst):
                writer.add_audio(f"(real){self.config.inst[i]}_hat", y_hat[0, i].cpu(), self.trainer.global_step , sample_rate=self.config.sr)
                
        if self.cerberus.transcription_head is not None:
            transcripts_hat = self.get_transcripts(mini_batch['mix'])
            for i in range(self.config.num_inst):
                writer.add_image(f"{self.config.inst[i]}_tr_hat", transcripts_hat[4, i].unsqueeze(0).cpu(), self.trainer.global_step)
                writer.add_image(f"{self.config.inst[i]}_tr_hat_{self.config.transcription_threshold}", (transcripts_hat[4, i].unsqueeze(0).cpu() > self.config.transcription_threshold).float(), self.trainer.global_step)
                if not self.saved_gt_to_tensorboard:
                    writer.add_image(f"{self.config.inst[i]}_tr_gt", mini_batch['transcripts_gt'][4, i].unsqueeze(0).cpu(), self.trainer.global_step)
            
            sample_transcription = self.get_transcripts(sample_audio)
            for i in range(self.config.num_inst):
                writer.add_image(f"(real){self.config.inst[i]}_tr_hat", sample_transcription[0, i].unsqueeze(0).cpu(), self.trainer.global_step)
                writer.add_image(f"(real){self.config.inst[i]}_tr_hat_{self.config.transcription_threshold}", (sample_transcription[0, i].unsqueeze(0).cpu() > self.config.transcription_threshold).float(), self.trainer.global_step)

        if not self.saved_gt_to_tensorboard:
            self.saved_gt_to_tensorboard = True


    def configure_optimizers(self):
        config = self.config

        if config.optimizer == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=self.lr)
        elif config.optimizer == "radam":
            from optimizer.radam import RAdam
            optimizer = RAdam(self.parameters(), lr=self.lr)
        elif config.optimizer == "rmsprop":
            optimizer = optim.RMSprop(self.parameters(), lr=self.lr)
        else:
            raise NotImplementedError
        
        reduce_on_plateau=False
        
        # Setting Scheduler
        monitor = None
        if config.lr_scheduler == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=200, eta_min=config.lr_min
            )

        elif config.lr_scheduler == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", patience=3, factor=config.lr_decay
            )
            monitor = "valid_tr_loss"
            reduce_on_plateau=True

        elif config.lr_scheduler == "multistep":
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                [5 * (x + 2) for x in range(500)],
                gamma=config.lr_decay,
            )

        elif config.lr_scheduler == "no":
            scheduler = None
        else:
            raise ValueError(f"unknown lr_scheduler :: {config.lr_scheduler}")

        if scheduler is not None:
            if monitor is not None:
                optimizers = [optimizer]
                schedulers = [
                    dict(
                        scheduler=scheduler,
                        monitor=monitor,
                        interval='epoch',
                        reduce_on_plateau=reduce_on_plateau,
                        frequency=config.check_val_every_n_epoch,
                    )]

                return optimizers, schedulers
            else:
                return [optimizer], [scheduler]
        else:
            return optimizer