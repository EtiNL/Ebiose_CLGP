import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=-1):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Normalize the embeddings to ensure they lie on the unit hypersphere
        output1 = F.normalize(output1, p=2, dim=1)
        output2 = F.normalize(output2, p=2, dim=1)

        cosine_similarity = F.cosine_similarity(output1, output2)
        
        loss_contrastive = torch.mean(
            (1 - label) * (cosine_similarity - self.margin).pow(2) +
            label * (1 - cosine_similarity).pow(2)
        )

        return loss_contrastive
