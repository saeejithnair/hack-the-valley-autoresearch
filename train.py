"""Agent-editable model, optimizer, and batch size. Evaluation stays fixed."""

import torch
from torch import nn

BATCH_SIZE = 32


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size: int, context_length: int):
        super().__init__()
        width = 64
        self.tokens = nn.Embedding(vocab_size, width)
        self.positions = nn.Embedding(context_length, width)
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=4, dim_feedforward=4 * width,
            dropout=0.0, batch_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.head = nn.Linear(width, vocab_size)
        self.register_buffer(
            "causal_mask", nn.Transformer.generate_square_subsequent_mask(context_length)
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(context.shape[1], device=context.device)
        hidden = self.tokens(context) + self.positions(positions)
        hidden = self.blocks(hidden, mask=self.causal_mask, is_causal=True)
        return self.head(hidden[:, -1])


def build_model(vocab_size: int, context_length: int) -> nn.Module:
    return TinyTransformer(vocab_size, context_length)


def build_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
