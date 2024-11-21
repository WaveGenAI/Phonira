import torch


class DelayPattern:
    def __init__(self, padding_value):
        self.padding_value = padding_value

    def reverse_pattern(self, x: torch.Tensor, start_pos: int = 0):
        b, k, n = x.shape
        assert b == 1, "Batch size must be 1, otherwise use stereo delay pattern"
        assert (
            start_pos <= n
        ), "Start position must be less or equal than the sequence length"

        out = x.clone()
        out[..., start_pos:] = torch.full_like(out[..., start_pos:], self.padding_value)

        for i in range(k):
            if i + start_pos >= n:
                break

            entry = x[:, i, i + start_pos : i + start_pos + out.shape[-1]]
            out[:, i, start_pos : start_pos + entry.size(-1)] = entry

        return out

    def apply_pattern(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """Apply the delayed pattern to the input tensor.

        Args:
            x (torch.Tensor): the input tensor
            start_pos (int, optional): the starting position. Defaults to 0.
        Returns:
            torch.Tensor: the delayed pattern tensor
        """

        b, cdbk, n = x.shape
        assert b == 1, "Batch size must be 1, otherwise use stereo delay pattern"
        assert (
            start_pos <= n
        ), "Start position must be less or equal than the sequence length"

        out = x.clone()
        out[:, :, start_pos:] = self.padding_value

        for k in range(min(cdbk, n)):
            out[:, k, start_pos + k : n] = x[:, k, start_pos : n - k]

        return out
