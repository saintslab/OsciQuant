import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from .parametrization import FakeQuantParametrization


def attach_weight_quantizers(model, exclude_layers, quantizer, enabled=True) -> None:
    """
    Attaches quantizers to a model in the form of parametrizations, to all layers except excluded.
    :param model: Model to be modified inplace
    :param exclude_layers: If a string from the list is in the layer name, layer will be excluded.
    :param quantizer: The quantizer to be used
    :param enabled: When True the quantizers is applied in the forward pass.
    """
    for name, module in model.named_modules():
        if not any(target in name for target in exclude_layers):
            if hasattr(module, "weight") and isinstance(module.weight, nn.Parameter):
                parametrize.register_parametrization(module, 'weight', FakeQuantParametrization(quantizer=quantizer, enabled=enabled))
                print(f"Attached weight quantizer to layer: {name}")


def detach_weight_quantizers(model, leave_parametrized=False) -> None:
    """
    Deataches the weight quantizers from a model.
    :param model: Model to be modified inplace
    :param leave_parametrized: If True the original weights is replaced with quantized weights, else original weights is kept 
    """
    for name, module in model.named_modules():
        if parametrize.is_parametrized(module, "weight"):
            parametrize.remove_parametrizations(module, "weight", leave_parametrized=leave_parametrized)
            print(f"Detached weight quantizer from layer: {name} ")


def toggle_quantization(model, enabled: bool) -> None:
    """
    Activates or deactivates the quantization of the weights. 
    For example can be done before eval to see quantized performance, while being disabled during training.
    :param model: Toggle quantizers for the given model 
    :param enabled: If quantizers is active or not
    """
    for name, submodule in model.named_modules():
        if hasattr(submodule, 'parametrizations'):
            # submodule.parametrizations is a dictionary like {"weight": [param_module, ...]}
            for param_name, param_list in submodule.parametrizations.items():
                for p in param_list:
                    if isinstance(p, FakeQuantParametrization):
                        p.enabled = enabled
