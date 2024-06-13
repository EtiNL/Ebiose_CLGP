import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data

class GCN(nn.Module):
    def __init__(self, config):
        super(GCN, self).__init__()
        
        in_channels, hidden_channels, out_channels = config.node_feature_context_length, config.graph_encoder.hidden, config.embed_dim
        
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.fc = nn.Linear(out_channels, out_channels)
        
    def initialize(self):
        # Initialize convolutional layers
        nn.init.xavier_uniform_(self.conv1.lin.weight)
        if self.conv1.lin.bias is not None:
            nn.init.zeros_(self.conv1.lin.bias)
        
        nn.init.xavier_uniform_(self.conv2.lin.weight)
        if self.conv2.lin.bias is not None:
            nn.init.zeros_(self.conv2.lin.bias)

        # Initialize the fully connected layer
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, graph_data):
        """
        Args:
            graph_data (Data): Graph data object containing x (node features), edge_index (graph connectivity),
                               and batch (batch vector).

        Returns:
            torch.tensor: embedding
        """
        x, edge_index, batch = graph_data.x, graph_data.edge_index, graph_data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return F.relu(x)
