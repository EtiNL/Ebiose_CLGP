import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.fc = nn.Linear(out_channels, out_channels)

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

# Example usage
if __name__ == "__main__":
    # Example graph data
    node_features = torch.tensor([[1, 2], [2, 3], [3, 4], [4, 5]], dtype=torch.float)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    # Create a Data object
    graph_data = Data(x=node_features, edge_index=edge_index, batch=batch)

    # Initialize the GCN model
    model = GCN(in_channels=2, hidden_channels=4, out_channels=2)

    # Forward pass
    embedding = model(graph_data)
    print(embedding)
