from collections import OrderedDict
import torch
from torch import nn

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, feedforward_dim: int, layer_norm_eps: float, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model, eps=layer_norm_eps)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, feedforward_dim)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(feedforward_dim, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model, eps=layer_norm_eps)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, config: dict):
        super().__init__()

        # Extract parameters from the configuration
        text_encoder_config = config['text_encoder']
        self.layers = text_encoder_config['layers']
        self.heads = text_encoder_config['heads']
        self.width = text_encoder_config['width']
        self.feedforward_dim = text_encoder_config['feedforward_dim']
        self.max_position_embeddings = text_encoder_config['max_position_embeddings']
        self.layer_norm_eps = text_encoder_config['layer_norm_eps']
        self.initializer_range = text_encoder_config['initializer_range']
        self.embed_dim = config['embed_dim']

        self.resblocks = nn.Sequential(*[
            ResidualAttentionBlock(self.width, self.heads, self.feedforward_dim, self.layer_norm_eps, 
                                   self.build_attention_mask(self.max_position_embeddings)) for _ in range(self.layers)])
        self.ln_final = LayerNorm(self.width, eps=self.layer_norm_eps)

        self.token_embedding = nn.Embedding(self.max_position_embeddings, self.width)
        self.positional_embedding = nn.Parameter(torch.empty(self.max_position_embeddings, self.width))
        self.text_projection = nn.Parameter(torch.empty(self.width, self.embed_dim))

        self.initialize()

    def build_attention_mask(self, context_length):
        mask = torch.empty(context_length, context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # Zero out the lower diagonal
        return mask

    def initialize(self):
        # Initialize the weights of the residual attention blocks
        for block in self.resblocks:
            nn.init.xavier_uniform_(block.attn.in_proj_weight)
            if block.attn.in_proj_bias is not None:
                nn.init.zeros_(block.attn.in_proj_bias)
            nn.init.xavier_uniform_(block.attn.out_proj.weight)
            if block.attn.out_proj.bias is not None:
                nn.init.zeros_(block.attn.out_proj.bias)

            nn.init.xavier_uniform_(block.mlp[0].weight)  # c_fc weight
            if block.mlp[0].bias is not None:
                nn.init.zeros_(block.mlp[0].bias)            # c_fc bias
            nn.init.xavier_uniform_(block.mlp[2].weight)  # c_proj weight
            if block.mlp[2].bias is not None:
                nn.init.zeros_(block.mlp[2].bias)            # c_proj bias

        # Initialize the text projection layer
        nn.init.xavier_uniform_(self.text_projection)

        # Initialize the token embedding
        nn.init.normal_(self.token_embedding.weight, std=self.initializer_range)

        # Initialize the positional embedding
        nn.init.normal_(self.positional_embedding, std=0.01)

    def forward(self, input_ids: torch.Tensor):
        # Embed tokens and positions
        token_embeddings = self.token_embedding(input_ids)
        position_ids = torch.arange(input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        position_embeddings = self.positional_embedding[position_ids]

        # Combine token and position embeddings
        x = token_embeddings + position_embeddings

        # Pass through residual attention blocks
        x = self.resblocks(x)

        # Apply final layer normalization
        x = self.ln_final(x)

        # Project to the embedding dimension
        x = torch.matmul(x, self.text_projection)

        return x