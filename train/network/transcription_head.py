import torch
import torch.nn as nn

class TranscriptionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_pitches = config.num_pitches
        self.num_inst = config.num_inst

        dense_in = config.lstm_hidden_size * (int(config.lstm_bidirectional)+1)
        dense_out = self.num_pitches * self.num_inst

        self.dense = nn.Linear(dense_in, dense_out)
        self.activate = nn.Sigmoid()

    def forward(self, shared_representation):
        # (batch, time, shared_representation)
        batch, time, embed_numinst = shared_representation.shape
        
        transcript = self.dense(shared_representation) # (batch, time, num_inst * pitches)
        transcript = self.activate(transcript)
        transcript = transcript.view(batch, time, self.num_inst, -1)
        transcript = transcript.permute(0, 2, 3, 1) # (batch, num_inst, pitch, time)

        return transcript