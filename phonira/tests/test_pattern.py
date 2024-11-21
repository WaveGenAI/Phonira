import random

import torch

from phonira.pattern import DelayPattern

pattern = DelayPattern(1024)


def test_full_delay_pattern():
    """Test the delay pattern function."""

    for _ in range(1, 1000):
        k = random.randint(1, 100)
        n = random.randint(k, 1000)
        x = torch.randint(0, 1024, (1, k, n))

        start_pos = random.randint(0, n + 10)

        try:
            out = pattern.apply_pattern(x, start_pos)
            out = pattern.reverse_pattern(out, start_pos)

            if start_pos >= n:
                raise ValueError(
                    "Start position is greater than the sequence length, should raise an error"
                )

        except Exception as e:
            if not (isinstance(e, AssertionError) and start_pos >= n):
                raise e
            else:
                return

        assert torch.all(
            x[..., : n - k] == out[..., : n - k]
        ), "The delay pattern is not correct"


def test_delay_pattern():
    """Test the delay pattern function."""

    a = torch.randint(0, 1023, [1, 9, 10])
    a[:, :, 0] = 1024

    a = pattern.apply_pattern(a, 1)
    a_test = pattern.reverse_pattern(a, 1)
    a_test = pattern.apply_pattern(a_test, 1)

    assert torch.all(a == a_test)
