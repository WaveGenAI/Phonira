import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """RMSNorm from https://arxiv.org/pdf/1910.07467"""

    def __init__(self, dim: int, epsilon: float = 1e-8):
        super().__init__()

        self.epsilon = epsilon
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.epsilon)
        x = x * self.g
        return x


class DecoderBlock(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()

        self.rms = RMSNorm(hidden_size)

    def forward(self, x):
        x_norm = self.rms(x)

        return x_norm


class Phonira(nn.Module):
    def __init__(self, num_quantizers: int, codebook_size: int, hidden_size: int):
        super().__init__()

        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(codebook_size + 1, hidden_size)  # +1 for padding
                for _ in range(num_quantizers)
            ]
        )

        self.decoder_blocks = nn.Sequential(
            *[DecoderBlock(hidden_size) for _ in range(1)]
        )

        self.dummy_model = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.heads = nn.ModuleList(
            [nn.Linear(hidden_size, codebook_size) for _ in range(num_quantizers)]
        )

    def forward(self, x, training=False):
        assert x.shape[1] == len(
            self.embeddings
        ), "Input shape mismatch with embeddings"

        if training:
            y = x[..., 1:]
            x = x[..., :-1]

        x = sum([embd(x[:, i, :]) for i, embd in enumerate(self.embeddings)])

        x = self.decoder_blocks(x)
        x = self.dummy_model(x)

        x = torch.stack([head(x) for head in self.heads], dim=1)

        if training:
            # compute cross entropy loss for each quantizer
            loss_fc = nn.CrossEntropyLoss()

            loss = 0
            for i in range(len(self.heads)):
                logits_loss = x[:, i].flatten(end_dim=1)
                target = y[:, i].flatten()

                loss += loss_fc(logits_loss, target)

            loss /= len(self.heads)

            return x, loss

        return x, None
