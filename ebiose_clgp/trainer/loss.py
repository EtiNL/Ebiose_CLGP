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
    
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1):  # Adjust the temperature as needed
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, output1, output2, labels):
        # Normalize the embeddings
        output1 = F.normalize(output1, p=2, dim=1)
        output2 = F.normalize(output2, p=2, dim=1)

        # Compute pairwise cosine similarity between output1 and output2
        similarity = torch.sum(output1 * output2, dim=1)  # [batch_size]

        # Ensure labels are on the same device as the similarity_matrix
        labels = labels.view(-1).cuda() if output1.is_cuda else labels.view(-1)
        
        # Debug print for label shape
        # print(f"Labels shape after reshape: {labels.shape}")
        # print(f"Labels: {labels}")

        # Compute logits
        logits = similarity / self.temperature  # [batch_size]
        
        # Compute loss using BCEWithLogitsLoss
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(logits, labels.float())
        
        return loss
