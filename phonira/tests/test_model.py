import torch
from torch import nn

from phonira.model import MultiHeadAttention, RMSNorm


def test_rmsnorm():
    input = torch.randn(2, 1500, 512)  # b, n, d
    _, _, d = input.size()

    rms_test = RMSNorm(d)
    rms_base = nn.RMSNorm(
        d, eps=1e-8
    )  # when single value is passed, it is considered as d

    assert torch.allclose(rms_test(input), rms_base(input)), "RMSNorm is incorrect"


def test_multiheadattention_one():
    embed_dim = 12
    input = torch.ones(1, 10, embed_dim)

    mha_test = MultiHeadAttention(embed_dim, 4)
    mha_base = nn.MultiheadAttention(embed_dim, 4)

    for param in mha_test.parameters():
        nn.init.constant_(param, 0.5)

    for param in mha_base.parameters():
        nn.init.constant_(param, 0.5)

    output_test = mha_test(input, input, input)
    output_base, _ = mha_base(input, input, input)

    assert torch.allclose(output_test, output_base, atol=1e-4), "Outputs are not equal"
