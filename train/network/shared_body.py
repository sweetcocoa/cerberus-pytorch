import torch
import torch.nn as nn
import torchaudio

class SharedBody(nn.Module):
    def __init__(self, config):
        super().__init__()
        input_size = config.n_fft//2 + 1 

        self.lstm = nn.LSTM(batch_first=True, 
                            input_size=input_size, 
                            hidden_size=config.lstm_hidden_size, 
                            num_layers=config.lstm_num_layers, 
                            bidirectional=config.lstm_bidirectional
                            )

        self.dropout = nn.Dropout(p=config.dropout_rate, inplace=False)

    def forward(self, spec) -> torch.Tensor:
        lstm_embed, (h, c) = self.lstm(spec)  # (batch, time, lstm_embedding_dim)
        shared_representation = self.dropout(lstm_embed)

        return shared_representation