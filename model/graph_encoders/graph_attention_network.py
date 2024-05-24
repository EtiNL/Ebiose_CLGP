import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1)
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
        # First graph attention convolution layer
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        
        # Second graph attention convolution layer
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        
        # Global mean pooling aggregates the node features to produce a single graph-level feature vector for each graph in the batch using the batch parameter.
        x = global_mean_pool(x, batch) 
        
        # Fully connected layer
        x = self.fc(x)
        
        return F.relu(x)  # Use ReLU activation to produce graph embeddings

# Example usage
if __name__ == "__main__":
    from torch_geometric.data import Data, DataLoader
    
    # Dummy data
    num_nodes = 6
    num_node_features = 3
    x = torch.randn((num_nodes, num_node_features), dtype=torch.float)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5],
                               [1, 2, 3, 4, 5, 0]], dtype=torch.long)  # Example edges
    batch = torch.tensor([0, 0, 1, 1, 1, 2], dtype=torch.long)  # Batch assignment

    # Create a Data object
    data = Data(x=x, edge_index=edge_index, batch=batch)

    # Model initialization
    model = GAT(in_channels=num_node_features, hidden_channels=8, out_channels=16, heads=2)

    # Forward pass
    graph_embeddings = model(data.x, data.edge_index, data.batch)
    print(graph_embeddings)
