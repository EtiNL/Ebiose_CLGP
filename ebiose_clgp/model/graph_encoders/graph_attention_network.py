import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class GAT(nn.Module):
    def __init__(self, config):
        super(GAT, self).__init__()

        in_channels, hidden_channels, out_channels = config.embed_dim, config.graph_encoder.hidden, config.embed_dim
        self.layers = config.graph_encoder.layers
        self.heads = config.graph_encoder.heads
        self.dropout = config.graph_encoder.dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        self.convs.append(GATConv(in_channels, hidden_channels, heads=self.heads))
        self.norms.append(nn.LayerNorm(hidden_channels * self.heads))

        for _ in range(self.layers - 2):
            self.convs.append(GATConv(hidden_channels * self.heads, hidden_channels, heads=self.heads))
            self.norms.append(nn.LayerNorm(hidden_channels * self.heads))
        
        self.convs.append(GATConv(hidden_channels * self.heads, out_channels, heads=1))
        self.norms.append(nn.LayerNorm(out_channels))

        self.fc = nn.Linear(out_channels, out_channels)

    def initialize(self):
        for conv in self.convs:
            nn.init.xavier_uniform_(conv.lin_src.weight)
            nn.init.xavier_uniform_(conv.lin_dst.weight)
            if conv.lin_src.bias is not None:
                nn.init.zeros_(conv.lin_src.bias)
            if conv.lin_dst.bias is not None:
                nn.init.zeros_(conv.lin_dst.bias)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, graph_data):
        x, edge_index, batch = graph_data.x, graph_data.edge_index, graph_data.batch

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        
        return F.relu(x)