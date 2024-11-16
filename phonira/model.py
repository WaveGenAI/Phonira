import torch
import torch.nn as nn


class Phonira(torch.nn.Module):
    def __init__(self, num_quantizers: int, codebook_size: int, hidden_size: int):
        super().__init__()

        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(codebook_size + 1, hidden_size)  # +1 for padding
                for _ in range(num_quantizers)
            ]
        )

    def forward(self, x, training=False):
        assert x.shape[1] == len(
            self.embeddings
        ), "Input shape mismatch with embeddings"

        if training:
            y = x[:, 1:]
            x = x[:, :-1]

        embds = sum([embd(x[:, i]) for i, embd in enumerate(self.embeddings)])

        return x, None
