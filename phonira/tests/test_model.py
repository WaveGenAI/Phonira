import torch
from torch import nn

from phonira.model import RMSNorm


def test_rmsnorm():
    input = torch.randn(2, 1500, 512)  # b, n, d
    _, _, d = input.size()

    rms_test = RMSNorm(d)
    rms_base = nn.RMSNorm(
        d, eps=1e-8
    )  # when single value is passed, it is considered as d

    assert torch.allclose(rms_test(input), rms_base(input)), "RMSNorm is incorrect"
