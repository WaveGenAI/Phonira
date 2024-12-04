import torch
import torch.nn as nn
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding
from torch.nn import functional as F
from tqdm import tqdm


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


class FFNSwiGLU(nn.Module):
    def __init__(self, dim: int, ff_dim: int, dropout_p: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(dim, ff_dim, bias=False)
        self.w2 = nn.Linear(ff_dim, dim, bias=False)

        self.v = nn.Linear(dim, ff_dim, bias=False)

        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        vx = self.v(x)

        x = F.silu(self.w1(x)) * vx
        x = self.dropout(x)
        return self.w2(x)


class MultiHeadAttention(nn.Module):
    """MultiHeadAttention from https://arxiv.org/pdf/1706.03762"""

    def __init__(self, hidden_size: int, num_heads: int, dropout_p: float = 0.1):
        super().__init__()
        self.w_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_v = nn.Linear(hidden_size, hidden_size, bias=False)

        self.w_out = nn.Linear(hidden_size, hidden_size, bias=False)

        # https://github.com/lucidrains/rotary-embedding-torch/issues/37
        self.rotary_emb = RotaryEmbedding(
            dim=hidden_size // num_heads, cache_if_possible=False
        )
        self.num_heads = num_heads
        self.dropout_p = dropout_p

    def forward(
        self,
        x_q,
        x_k,
        x_v,
        padding_mask: torch.Tensor = None,
        is_causal: bool = False,
        start_pos: int = 0,
    ):
        q = self.w_q(x_q)
        k = self.w_k(x_k)
        v = self.w_v(x_v)

        q_head = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k_head = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v_head = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)

        # apply rotary embeddings
        q_head = self.rotary_emb.rotate_queries_or_keys(q_head)
        k_head = self.rotary_emb.rotate_queries_or_keys(k_head)

        b, _, n, _ = q_head.shape

        # create attention mask
        attention_mask = torch.ones((b, 1, n, n), device=x_q.device, dtype=torch.bool)

        if is_causal:
            # create a causal mask using the start_pos to avoid masking the prepend tokens
            attention_mask[:, :, :start_pos, start_pos:] = False
            attention_mask[:, :, start_pos:, start_pos:] = torch.tril(
                attention_mask[:, :, start_pos:, start_pos:]
            )

        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(1)
            padding_mask = padding_mask.expand(b, 1, n, n)
            attention_mask = attention_mask & padding_mask

        dropout_p = 0
        if self.training:
            dropout_p = self.dropout_p

        # pylint: disable=E1102
        attention = F.scaled_dot_product_attention(
            q_head, k_head, v_head, attn_mask=attention_mask, dropout_p=dropout_p
        )

        out = rearrange(attention, "b h n d -> b n (h d)")
        out = self.w_out(out)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout_p: float = 0.1):
        super().__init__()

        self.rms1 = RMSNorm(hidden_size)
        self.rms2 = RMSNorm(hidden_size)

        self.dropout = nn.Dropout(dropout_p)

        self.mha = MultiHeadAttention(hidden_size, num_heads, dropout_p)

        ff_dim = int((hidden_size * 4) * (2 / 3))
        self.feed_forward = FFNSwiGLU(hidden_size, ff_dim, dropout_p)

    def forward(self, x, padding_mask: torch.Tensor = None, start_pos: int = 0):
        x_bis = self.rms1(x)
        x_bis = self.mha(
            x_bis,
            x_bis,
            x_bis,
            padding_mask=padding_mask,
            is_causal=True,
            start_pos=start_pos,
        )
        x_bis = self.dropout(x_bis)
        x = x + x_bis

        x_bis = self.rms2(x)
        x_bis = self.feed_forward(x_bis)
        x_bis = self.dropout(x_bis)
        x = x + x_bis
        return x


class Phonira(nn.Module):
    def __init__(
        self,
        num_quantizers: int,
        codebook_size: int,
        hidden_size: int,
        depth: int,
        padding_token: int,
        proj_dim: int,
        delay_pattern,
        num_heads: int,
        dropout_p: float = 0.1,
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
            [
                DecoderBlock(hidden_size, num_heads=num_heads, dropout_p=dropout_p)
                for _ in range(depth)
            ]
        )

        self.heads = nn.ModuleList(
            [
                nn.Linear(hidden_size, codebook_size, bias=False)
                for _ in range(num_quantizers)
            ]
        )

        self.rms = RMSNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_p)

        if proj_dim != hidden_size:
            self.proj = nn.Linear(proj_dim, hidden_size, bias=False)
        else:
            self.proj = nn.Identity()

        self.delay_pattern = delay_pattern

    def forward(
        self,
        x,
        prepend_embed: torch.Tensor = None,
        padding_mask: torch.Tensor = None,
        prepend_mask: torch.Tensor = None,
        training=False,
    ):
        assert x.shape[1] == len(
            self.embeddings
        ), "Input shape mismatch with embeddings"

        if training:
            y = x[..., 1:]
            x = x[..., :-1]

        start_pos = 0
        x = sum([embd(x[:, i, :]) for i, embd in enumerate(self.embeddings)])
        x = self.dropout(x)

        if prepend_embed is not None:
            prepend_embed = self.proj(prepend_embed)

            # prepend embeddings
            start_pos = prepend_embed.size(1)
            padding_mask = torch.cat([prepend_mask, padding_mask], dim=1)
            x = torch.cat([prepend_embed, x], dim=1)

        if training:
            padding_mask = padding_mask[:, :-1]

        for block in self.decoder_blocks:
            x = block(x, padding_mask=padding_mask, start_pos=start_pos)

        x = self.rms(x)

        x = torch.stack([head(x) for head in self.heads], dim=1)

        if training:
            # compute cross entropy loss for each quantizer
            loss_fc = nn.CrossEntropyLoss()

            loss = 0
            for i in range(len(self.heads)):
                # get the last token of x (without the prepend token)
                logits_loss = x[:, i, -y.size(-1) :, :].flatten(end_dim=1)
                target = y[:, i].flatten()

                # replace padding token with -100
                target = torch.where(target == self._pad_token, -100, target)

                loss += loss_fc(logits_loss, target)

            loss /= len(self.heads)

            return x, loss

        return x, None

    def generate(
        self,
        prepend_embed: torch.Tensor,
        prepend_mask: torch.Tensor,
        num_quantizers: int,
        num_gen: int,
        padding_value: int = 1024,
        temperature: int = 1.0,
        top_k: int = 150,
        guidance_scale: float = 3.0,
    ):
        assert (
            num_gen >= num_quantizers
        ), "num_gen must be greater or equal to num_quantizers"

        x = torch.fill_(
            torch.empty(
                1, num_quantizers, 1, dtype=torch.long, device=prepend_embed.device
            ),
            padding_value,
        )
        for _ in tqdm(range(num_gen), desc="Generating sample"):
            x = self.delay_pattern.apply_pattern(x, start_pos=1)
            padding_mask = torch.ones_like(x[:, 0, :]).bool()

            out = self.forward(x, prepend_embed, padding_mask, prepend_mask)[0]
            out = out[:, :, -1, :]

            out_uncond = self.forward(x, None, padding_mask, None)[0]
            out_uncond = out_uncond[:, :, -1, :]

            out = out / temperature
            out_uncond = out_uncond / temperature

            out = F.log_softmax(out, dim=-1)
            out_uncond = F.log_softmax(out_uncond, dim=-1)

            # Apply classifier-free guidance
            out = out_uncond + guidance_scale * (out - out_uncond)

            # convert to probabilities
            out = F.softmax(out, dim=-1)

            topk_tokens, indices = out.topk(top_k, dim=-1)
            topk_tokens = topk_tokens.view(-1, top_k)

            samples = torch.multinomial(topk_tokens, 1).unsqueeze(0)
            indices = torch.gather(indices, -1, samples)

            x = torch.cat([x, indices], dim=-1)
            x = self.delay_pattern.reverse_pattern(x, start_pos=1)

        return x[..., 1 : x.size(-1) - num_quantizers + 1]
