import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool

class GIN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GIN, self).__init__()
        nn1 = nn.Sequential(nn.Linear(in_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels))
        nn2 = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, out_channels))
        self.conv1 = GINConv(nn1)
        self.conv2 = GINConv(nn2)
        self.fc = nn.Linear(out_channels, out_channels)

    def forward(self, x, edge_index, batch):
        """_summary_

        Args:
            x (torch.tensor): Node feature matrix of shape [num_nodes, num_node_features]
            edge_index (torch.tensor): Graph connectivity in COO format (i.e., a list of edge indices) of shape [2, num_edges]
            batch (torch.tensor): Batch vector which assigns each node to a specific graph in the batch of shape [num_nodes]
                                  Example: If the first two nodes belong to the first graph, the next three to the second graph, and the last node to the third graph, 
                                  batch might look like:

                                  batch = torch.tensor([0, 0, 1, 1, 1, 2], dtype=torch.long)

        Returns:
            torch.tensor: embedding
        """
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_add_pool(x, batch)
        x = self.fc(x)
        return F.relu(x)
