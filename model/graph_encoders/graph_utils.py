import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data

def parse_graph(graph_struct):
    # Create a NetworkX graph
    G = nx.Graph()

    # Add nodes with features
    G.add_node(0, features=[1, 0, 0])
    G.add_node(1, features=[0, 1, 0])
    G.add_node(2, features=[0, 0, 1])

    # Add edges
    G.add_edge(0, 1)
    G.add_edge(1, 2)
    
    return G

def graph_struct_to_graph_data(graph_struct):
    
    G = parse_graph(graph_struct)

    # Extract node features
    node_features = np.array([G.nodes[i]['features'] for i in G.nodes])

    # Convert to PyTorch tensors
    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(np.array(G.edges).T, dtype=torch.long)

    # Create a PyTorch Geometric data object
    graph_data = Data(x=x, edge_index=edge_index)
    
    return graph_data