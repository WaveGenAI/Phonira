import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F


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


class MultiHeadAttention(nn.Module):
    """MultiHeadAttention from https://arxiv.org/pdf/1706.03762"""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.w_q = nn.Linear(hidden_size, hidden_size)
        self.w_k = nn.Linear(hidden_size, hidden_size)
        self.w_v = nn.Linear(hidden_size, hidden_size)

        self.w_out = nn.Linear(hidden_size, hidden_size)

        self.num_heads = num_heads

    def forward(
        self, x_q, x_k, x_v, padding_mask: torch.Tensor = None, is_causal: bool = True
    ):
        q = self.w_q(x_q)
        k = self.w_k(x_k)
        v = self.w_v(x_v)

        q_head = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k_head = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v_head = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)

        b, _, n, _ = q_head.shape

        # create attention mask
        attention_mask = torch.ones((b, 1, n, n), device=x_q.device, dtype=torch.bool)

        if is_causal:
            attention_mask = torch.tril(attention_mask)

        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(1)
            padding_mask = padding_mask.expand(b, 1, n, n)
            attention_mask = attention_mask & padding_mask

        attention = F.scaled_dot_product_attention(
            q_head, k_head, v_head, attn_mask=attention_mask
        )

        out = rearrange(attention, "b h n d -> b n (h d)")
        out = self.w_out(out)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()

        self.rms1 = RMSNorm(hidden_size)
        self.rms2 = RMSNorm(hidden_size)

        self.mha = MultiHeadAttention(hidden_size, 8)

    def forward(self, x, padding_mask: torch.Tensor = None):
        x_norm = self.rms1(x)
        x = self.mha(x_norm, x_norm, x_norm, padding_mask=padding_mask)

        x = x + x_norm

        return x


class Phonira(nn.Module):
    def __init__(
        self,
        num_quantizers: int,
        codebook_size: int,
        hidden_size: int,
        padding_token: int,
    ):
        super().__init__()
        self._pad_token = padding_token

        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(codebook_size + 1, hidden_size)  # +1 for padding
                for _ in range(num_quantizers)
            ]
        )

        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(hidden_size) for _ in range(1)]
        )

        self.heads = nn.ModuleList(
            [nn.Linear(hidden_size, codebook_size) for _ in range(num_quantizers)]
        )

    def forward(self, x, padding_mask: torch.Tensor = None, training=False):
        assert x.shape[1] == len(
            self.embeddings
        ), "Input shape mismatch with embeddings"

        if training:
            y = x[..., 1:]
            x = x[..., :-1]

        x = sum([embd(x[:, i, :]) for i, embd in enumerate(self.embeddings)])

        for block in self.decoder_blocks:
            x = block(x, padding_mask=padding_mask[..., :-1])

        x = torch.stack([head(x) for head in self.heads], dim=1)

        if training:
            # compute cross entropy loss for each quantizer
            loss_fc = nn.CrossEntropyLoss()

            loss = 0
            for i in range(len(self.heads)):
                logits_loss = x[:, i].flatten(end_dim=1)
                target = y[:, i].flatten()

                # replace padding token with -100
                target[target == self._pad_token] = -100

                loss += loss_fc(logits_loss, target)

            loss /= len(self.heads)

            return x, loss

        return x, None
