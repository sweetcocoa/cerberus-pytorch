import torch

def apply_masks(mask, mix_mag, num_inst):
    mix_shape = mix_mag.shape
    mix_mag_repeat = mix_mag.repeat_interleave(num_inst, 0).view(mix_shape[0], num_inst, mix_shape[1], mix_shape[2])
    sep_mag_hat = mask * mix_mag_repeat
    return sep_mag_hat

def realimag(mag, phase):
    """
    Combine a magnitude spectrogram and a phase spectrogram to a complex-valued spectrogram with shape (*, 2)
    """
    spec_real = mag * torch.cos(phase)
    spec_imag = mag * torch.sin(phase)
    spec = torch.stack([spec_real, spec_imag], dim=-1)
    return spec