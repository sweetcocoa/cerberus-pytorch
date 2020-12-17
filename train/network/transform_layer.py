import torch
import torch.nn as nn
import torchaudio

class ISTFT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hop_length = config.hop_length
        self.n_fft = config.n_fft
        window = torch.hann_window(config.n_fft)
        self.register_buffer("window", window)
    
    def forward(self, X : torch.Tensor):

        num_inst = 1
        if len(X.shape) == 5:
            # 악기별 X일 때 (batch, inst, freq, time, 2)
            num_inst = X.shape[1]
            X = X.view(X.shape[0]*X.shape[1], X.shape[2], X.shape[3], 2)
            
        x = torch.istft(X, 
                        n_fft=self.n_fft, 
                        hop_length=self.hop_length, 
                        window=self.window, 
                        return_complex=False
                        )
        if num_inst != 1:
            x = x.view(x.shape[0] // num_inst, num_inst, -1)
        
        return x