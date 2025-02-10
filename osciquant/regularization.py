import torch
import torch.nn as nn
from .parametrization import FakeQuantParametrization


class OsciQuantLoss(nn.Module):
    """
    Regularizes all layers which have a FakeQuantParametrization on their weight
    """
    def __init__(self, base_loss, model, regularization_lambda=1.0, regularization=True):
        super(OsciQuantLoss, self).__init__()
        self.base_loss = base_loss
        self.model = model
        self.regularization_lambda = regularization_lambda
        self.regularization = regularization

    def forward(self, output, target):
        # Base loss
        loss = self.base_loss(output, target)

        # Regularization when training
        if self.regularization and self.model.training:
            regularization_term = 0.0
            for name, module in self.model.named_modules():
                if hasattr(module, "parametrizations") and "weight" in module.parametrizations:
                    for param_module in module.parametrizations["weight"]:
                        if isinstance(param_module, FakeQuantParametrization):
                            # Compute regularization term
                            W_float = module.parametrizations.weight.original
                            W_quant = param_module.quantize_weights(W_float)
                            negative_delta = W_quant**2 - W_float**2
                            regularization_term += torch.mean(negative_delta)

            loss = loss + self.regularization_lambda * regularization_term

        return loss
