import torch
from torch_geometric.data import Data

def combine_graphs(graph_data_list):
    """
    Combines multiple graph data tuples into a single Data object with a batch attribute.

    Args:
        graph_data_list (list of tuples): List of tuples, where each tuple contains
                                          (node_features, edge_index) as torch.Tensor.

    Returns:
        Data: Combined Data object with node features, edge indices, and batch attribute.
    """
    # Initialize lists to collect node features, edge indices, and batch information
    all_node_features = []
    all_edge_indices = []
    batch = []

    # Offset for edge indices
    node_offset = 0

    for i, (node_features, edge_index) in enumerate(graph_data_list):
        num_nodes = node_features.size(0)

        # Append node features
        all_node_features.append(node_features)

        # Adjust edge indices with the current offset and append
        edge_index_adjusted = edge_index + node_offset
        all_edge_indices.append(edge_index_adjusted)

        # Append batch information
        batch.append(torch.full((num_nodes,), i, dtype=torch.long))

        # Update the offset for the next graph
        node_offset += num_nodes

    # Concatenate all collected data
    combined_node_features = torch.cat(all_node_features, dim=0)
    combined_edge_indices = torch.cat(all_edge_indices, dim=1)
    combined_batch = torch.cat(batch, dim=0)

    # Create and return a combined Data object
    combined_graph = Data(x=combined_node_features, edge_index=combined_edge_indices, batch=combined_batch)
    return combined_graph
