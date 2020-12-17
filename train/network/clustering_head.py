import torch
import torch.nn as nn

class ClusteringHead(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Deep cluster head
        dense_in = config.lstm_hidden_size * (int(config.lstm_bidirectional)+1)
        dense_out = (config.n_fft//2 + 1) * (config.embedding_size)

        self.num_inst = config.num_inst
        self.embedding_size = config.embedding_size
        
        self.dense_embedding = nn.Linear(dense_in, dense_out)
        self.activate = nn.Tanh()

    
    def forward(self, shared_representation: torch.Tensor) -> torch.Tensor:
        # (batch, time, shared_representation)

        batch, time, embed_numinst = shared_representation.shape
        
        proj = self.dense_embedding(shared_representation) # (batch, time, embedding_size * n_frequency)
        proj = self.activate(proj)
        
        proj = proj.view(batch, -1, self.embedding_size) # (batch, time * n_frequency, embedding_size)
        proj_norm = torch.norm(proj, p=2, dim=-1, keepdim=True)
        proj_one = proj / (proj_norm + 1e-12)

        return proj_one
