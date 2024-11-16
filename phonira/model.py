import torch


class Phonira(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = torch.nn.Embedding(100, 100)

    def forward(self, x, training=False):
        return x, None
