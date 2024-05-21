# adapted code from https://github.com/ngthanhtin/CLIP-Training-Pytorch

from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class CLGP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # graph
                 Graph_encoder,
                 graph_encoder_config,
                 # text
                 Text_encoder,
                 text_encoder_config,
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int):
        super().__init__()

        self.context_length = context_length

        self.graph_encoder = Graph_encoder(graph_encoder_config)

        self.text_encoder = Text_encoder(text_encoder_config)

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        #initialise graph_encoder
        self.graph_encoder.initialize()

        #initialise text_encoder
        self.text_encoder.initialize(self.text_projection)


    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_graph(self, graph):
        return self.gnn(graph.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_encoder(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        graph_features = self.encode_graph(image)
        text_features = self.encode_text(text)

        return graph_features, text_features 