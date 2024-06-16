import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GCN(nn.Module):
    def __init__(self, config):
        super(GCN, self).__init__()
        
        in_channels, hidden_channels, out_channels = config.embed_dim, config.graph_encoder.hidden, config.embed_dim
        self.layers = config.graph_encoder.layers
        
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        
        for _ in range(self.layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.fc = nn.Linear(out_channels, out_channels)
        
    def initialize(self):
        # Initialize convolutional layers
        for conv in self.convs:
            nn.init.xavier_uniform_(conv.lin.weight)
            if conv.lin.bias is not None:
                nn.init.zeros_(conv.lin.bias)

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
        
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        
        return F.relu(x)
