import torch
import torch.nn as nn


class DeepClusterLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tf_embedding: torch.Tensor, separation_gt_spec: torch.Tensor) -> torch.Tensor:
        """
        tf_embedding : (batch, freq*time, embedding_size)
        separation_gt_spec : (batch, inst, time, freq)
        """
        batch, num_inst, time, freq = separation_gt_spec.shape

        target_idx = separation_gt_spec.argmax(dim=1)
        t = nn.functional.one_hot(target_idx, num_classes=num_inst).float()
        t = t.view(batch, time*freq, num_inst)
        
        v = tf_embedding
        
        # (b, embedding_size, TF) *  (b, TF, embedding_size) = (b, embedding_size, embedding_size)
        vvT = torch.matmul(v.transpose(-1, -2), v)
        ttT = torch.matmul(t.transpose(-1, -2), t)
        vTt = torch.matmul(v.transpose(-1, -2), t)

        loss = vvT.norm() + ttT.norm() - 2*vTt.norm()
        return loss