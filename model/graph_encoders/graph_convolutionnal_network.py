import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import yaml

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class GCNEncoder(torch.nn.Module):
    def __init__(self, config_file, embed_dim: int):
        super(GCNEncoder, self).__init__()
        # Open and read the YAML file
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
            
        self.width = config['width']
        self.layers = config['layers']
        self.heads = config['heads']
        self.context_length = context_length
        
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(self.width, self.heads, self.build_attention_mask(self.context_length)) for _ in range(self.layers)])
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x