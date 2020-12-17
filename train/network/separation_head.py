import torch
import torch.nn as nn
import torchaudio

class SeparationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Mask heads
        dense_in = config.lstm_hidden_size * (int(config.lstm_bidirectional)+1)
        dense_out = (config.n_fft//2 + 1) * config.num_inst

        self.num_inst = config.num_inst
        self.dense = nn.Linear(dense_in, dense_out)
        self.activate = nn.Softmax2d()


    def forward(self, shared_representation: torch.Tensor) -> torch.Tensor:
        # (batch, time, shared_representation)
        
        mask = self.dense(shared_representation)  # (batch, time, frequency * num_inst) mask
        batch, time, frequency_inst = mask.shape
        mask = mask.view(batch, time, self.num_inst, -1) # (batch, time, num_inst, frequency)
        masks = mask.permute(0, 2, 1, 3)  # (batch, inst, time, freq)
        masks = self.activate(masks)

        return masks