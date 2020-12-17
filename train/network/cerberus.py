import torch
import torch.nn as nn
import torchaudio
import torchaudio.models

from network.shared_body import SharedBody
from network.clustering_head import ClusteringHead
from network.separation_head import SeparationHead
from network.transcription_head import TranscriptionHead

class Cerberus(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.shared_body = SharedBody(config)
        if 'sep' in self.config.heads:
            self.separation_head = SeparationHead(config)
        else:
            self.separation_head = None

        if 'dc' in self.config.heads:
            self.clustering_head = ClusteringHead(config)
        else:
            self.clustering_head = None

        if 'tr' in self.config.heads:
            self.transcription_head = TranscriptionHead(config)
        else: 
            self.transcription_head = None

    def forward(self, mix_mag):
        spec = mix_mag
        shared_representation = self.shared_body(spec)

        rt = dict()

        if self.separation_head is not None:
            separation_mask = self.separation_head(shared_representation)
            rt['separation_mask'] = separation_mask

        if self.clustering_head is not None:
            embedding = self.clustering_head(shared_representation)
            rt['embedding'] = embedding
        
        if self.transcription_head is not None:
            transcripts = self.transcription_head(shared_representation)
            rt['transcripts'] = transcripts

        return rt