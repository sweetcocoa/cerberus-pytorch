import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F
from utils.dsp import apply_masks

class MaskInferenceLoss(nn.Module):
    def __init__(self, num_inst):
        super().__init__()
        self.num_inst = num_inst
        self.mse = nn.MSELoss()

    def forward(self, separation_mask, mix_mag, separation_gt_mag):
        """
        separation_mask : (batch, inst, time, freq)
        mix_mag : (batch, time, freq)
        separation_gt_mag : (batch, inst, time, freq)
        """
        hat = apply_masks(separation_mask, mix_mag, self.num_inst)
        loss = self.mse(hat, separation_gt_mag)
        return loss