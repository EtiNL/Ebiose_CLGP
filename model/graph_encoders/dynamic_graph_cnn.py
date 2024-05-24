import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DynamicEdgeConv, global_max_pool

class DGCNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, k=20):
        super(DGCNN, self).__init__()
        self.conv1 = DynamicEdgeConv(nn.Sequential(nn.Linear(2 * in_channels, hidden_channels), nn.ReLU()), k=k)
        self.conv2 = DynamicEdgeConv(nn.Sequential(nn.Linear(2 * hidden_channels, hidden_channels), nn.ReLU()), k=k)
        self.conv3 = DynamicEdgeConv(nn.Sequential(nn.Linear(2 * hidden_channels, out_channels), nn.ReLU()), k=k)
        self.fc = nn.Linear(out_channels, out_channels)

    def forward(self, x, batch):
        """_summary_

        Args:
            x (torch.tensor): Node feature matrix
            batch (torch.tensor): Batch vector which assigns each node to a specific graph in the batch.

        Returns:
            embbeding (torch.tensor)
        """
        x = self.conv1(x, batch)
        x = self.conv2(x, batch)
        x = self.conv3(x, batch)
        x = global_max_pool(x, batch)
        x = self.fc(x)
        return F.relu(x)
