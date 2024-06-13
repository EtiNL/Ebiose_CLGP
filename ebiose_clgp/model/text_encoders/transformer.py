from collections import OrderedDict
import torch
from torch import nn
import yaml

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
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, embed_dim: int, context_length: int, width: int, layers: int, heads: int):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.width = width
        self.layers = layers
        self.heads = heads
        self.context_length = context_length
        
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(self.width, self.heads, self.build_attention_mask(self.context_length)) for _ in range(self.layers)])
        self.ln_final = LayerNorm(self.width)
        
    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask
    
    def initialize(self):
        # Initialize the weights of the residual attention blocks
        for block in self.resblocks:
            nn.init.xavier_uniform_(block.attn.in_proj_weight)
            nn.init.zeros_(block.attn.in_proj_bias)
            nn.init.xavier_uniform_(block.attn.out_proj.weight)
            nn.init.zeros_(block.attn.out_proj.bias)

            nn.init.xavier_uniform_(block.mlp[0].weight)  # c_fc weight
            nn.init.zeros_(block.mlp[0].bias)            # c_fc bias
            nn.init.xavier_uniform_(block.mlp[2].weight)  # c_proj weight
            nn.init.zeros_(block.mlp[2].bias)            # c_proj bias

        # Initialize the text projection layer
        nn.init.xavier_uniform_(self.text_projection)

        # Initialize the token embedding
        nn.init.normal_(self.token_embedding.weight, std=0.02)

        # Initialize the positional embedding
        nn.init.normal_(self.positional_embedding, std=0.01)
    
    def forward(self, text_data):
        x = text_data.type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.resblocks(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text_data.argmax(dim=-1)] @ self.text_projection

        return x