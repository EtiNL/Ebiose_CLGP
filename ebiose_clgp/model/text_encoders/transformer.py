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

        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(self.width, self.heads, self.feedforward_dim, self.layer_norm_eps, self.build_attention_mask(self.max_position_embeddings)) for _ in range(self.layers)])
        self.ln_final = LayerNorm(self.width, eps=self.layer_norm_eps)

        self.token_embedding = nn.Embedding(self.max_position_embeddings, self.width)
        self.positional_embedding = nn.Parameter(torch.empty(self.max_position_embeddings, self.width))
        self.text_projection = nn.Parameter(torch.empty(self.width, self.embed_dim))

        self.initialize()

    def build_attention_mask(self, context_length):
        # Lazily create causal attention mask, with full attention between the vision tokens
        # PyTorch uses additive attention mask; fill with -inf
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
        # Check dimensions of text_data
        print("text_data shape:", text_data.shape)  # Debugging line
        assert text_data.size(1) <= self.max_position_embeddings, "text_data length exceeds max_position_embeddings"

        x = self.token_embedding(text_data)  # [batch_size, context_length, width]
        print("x shape after token_embedding:", x.shape)  # Debugging line

        # Ensure positional_embedding dimensions are compatible
        positional_embedding = self.positional_embedding[:x.size(1), :].unsqueeze(0)
        print("positional_embedding shape after unsqueeze:", positional_embedding.shape)  # Debugging line
        positional_embedding = positional_embedding.expand(x.size(0), -1, -1)
        print("positional_embedding shape after expand:", positional_embedding.shape)  # Debugging line
        positional_embedding = positional_embedding.to(x.device)
        x = x + positional_embedding

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.resblocks(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        
        # x.shape = [batch_size, context_length, transformer.width]
        # Take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text_data.argmax(dim=-1)] @ self.text_projection

        return x