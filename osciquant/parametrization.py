import torch
import torch.nn as nn


class FakeQuantParametrization(nn.Module):
    """
    Parametrization module that applies fake quant if `enabled=True`.
    Otherwise, returns the original weights unchanged.
    """

    def __init__(self, quantizer, enabled=True):
        super().__init__()
        self.quantizer = quantizer
        self.enabled = enabled

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return W
        return self.quantize_weights(W)

    def quantize_weights(self, W: torch.Tensor) -> torch.Tensor:
        return self.quantizer(W)
