import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Ensure that output1, output2, and label have the correct shapes
        assert output1.shape == output2.shape, f"Shape mismatch: output1 shape {output1.shape} != output2 shape {output2.shape}"
        assert output1.shape[0] == label.shape[0], f"Batch size mismatch: output1 shape {output1.shape[0]} != label shape {label.shape[0]}"
        assert len(label.shape) == 1 or (len(label.shape) == 2 and label.shape[1] == 1), f"Label shape must be (batch_size,) or (batch_size, 1), but got {label.shape}"

        # Calculate the Euclidean distance between the two output embeddings
        euclidean_distance = F.pairwise_distance(output1, output2)

        # Calculate the contrastive loss
        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2) +
            (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )

        return loss_contrastive