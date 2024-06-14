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

    def forward(self, text_data):
        print("text_data shape:", text_data.shape)  # Debugging line
        assert text_data.size(1) <= self.max_position_embeddings, "text_data length exceeds max_position_embeddings"

        x = self.token_embedding(text_data)  # [batch_size, context_length, width]
        print("x shape after token_embedding:", x.shape)  # Debugging line

        # Ensure positional_embedding dimensions are compatible
        positional_embedding = self.positional_embedding[:x.size(1), :].unsqueeze(0)
        print("positional_embedding shape after unsqueeze:", positional_embedding.shape)  # Debugging line

        # Check if positional_embedding is on the same device as x
        print(f"x device: {x.device}, positional_embedding device: {positional_embedding.device}")
        positional_embedding = positional_embedding.to(x.device)
        print("positional_embedding shape after to device:", positional_embedding.shape)  # Debugging line

        # Use broadcasting instead of repeat
        positional_embedding = positional_embedding.expand(x.size(0), -1, -1)
        print("positional_embedding shape after expand:", positional_embedding.shape)  # Debugging line

        # Ensure that dimensions match before adding
        assert x.shape == positional_embedding.shape, f"x shape: {x.shape}, positional_embedding shape: {positional_embedding.shape}"
        try:
            x = x + positional_embedding
        except RuntimeError as e:
            print(f"Error during addition: {e}")
            print(f"x shape: {x.shape}, positional_embedding shape: {positional_embedding.shape}")
            raise

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.resblocks(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, context_length, transformer.width]
        # Take features from the eot embedding (eot_token is the highest number in each sequence)
        eot_indices = text_data.argmax(dim=-1)
        print("eot_indices:", eot_indices)  # Debugging line

        # Ensure eot_indices are within bounds
        assert torch.all(eot_indices < x.size(1)), f"eot_indices: {eot_indices}, context_length: {x.size(1)}"
        x = x[torch.arange(x.shape[0]), eot_indices] @ self.text_projection

        return x