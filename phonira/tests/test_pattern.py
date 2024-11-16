import random

import torch

from phonira.utils import delay_pattern, reverse_delay_pattern

# valid example
x = torch.tensor(
    [
        [
            [186, 822, 68, 878, 212, 130, 802, 473, 371, 582, 544, 813],
            [599, 378, 791, 259, 91, 170, 724, 793, 113, 290, 80, 57],
            [409, 454, 290, 141, 914, 766, 325, 335, 535, 420, 158, 628],
            [15, 805, 967, 938, 814, 390, 684, 520, 43, 754, 966, 649],
            [365, 47, 913, 640, 81, 939, 42, 655, 588, 924, 782, 306],
        ]
    ],
)


y = torch.tensor(
    [
        [
            [186, 822, 68, 878, 212, 130, 802, 473, 371, 582, 544, 813],
            [1025, 599, 378, 791, 259, 91, 170, 724, 793, 113, 290, 80],
            [1025, 1025, 409, 454, 290, 141, 914, 766, 325, 335, 535, 420],
            [1025, 1025, 1025, 15, 805, 967, 938, 814, 390, 684, 520, 43],
            [1025, 1025, 1025, 1025, 365, 47, 913, 640, 81, 939, 42, 655],
        ]
    ]
)


def test_full_delay_pattern():
    """Test the delay pattern function."""

    for _ in range(1, 1000):
        k = random.randint(1, 100)
        n = random.randint(k, 1000)
        x = torch.randint(0, 1024, (1, k, n))
        out = delay_pattern(x)
        out = reverse_delay_pattern(out)
        assert torch.all(
            x[..., : out.shape[-1]] == out
        ), "The delay pattern is not correct"


def test_delay_pattern():
    """Test the delay pattern function."""
    output_tensor = delay_pattern(x)
    assert torch.all(output_tensor == y), "The delay pattern is not correct"
