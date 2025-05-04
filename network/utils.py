import torch
import torch.nn as nn
from thop import profile


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample.
    """
    if drop_prob == 0.0 or not training:
        return x

    keep_prob = 1 - drop_prob
    # Generate binary mask
    shape = (x.size(0),) + (1,) * (x.dim() - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype = x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor

class DropPath(nn.Module):
    """
    Stochastic Depth: randomly drops whole residual branches.
    A regularization trick where, during training, entire residual branches are randomly dropped with some probability.
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)
    
def count_parameters(model: nn.Module, only_trainable: bool = True) -> int:
    """
    Count parameters in a model.
    Args:
        model: any torch.nn.Module
        only_trainable: if True, only counts parameters with requires_grad=True
    Returns:
        total number of parameters
    """
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

def print_model_size(model: nn.Module, only_trainable: bool = True) -> None:
    """print total parameters (in millions)."""
    total = count_parameters(model, only_trainable)
    print(f"Total Params: {total/1e6:.2f} M ({'trainable' if only_trainable else 'all'})")

def count_flops(model: nn.Module, input_size: tuple, as_string:bool = True, verbose=False):
    """
    Estimate FLOPs given a dummy input shape.
    Args:
        model: nn.Module
        input_size: e.g. (1, 3, 32, 32)
        as_string: if True, prints a human-readable summary
        verbose: if True, return THOPs layer-by-layer breakdown for debugging
    Returns:
        flops, params (both as raw integers if as_string=False)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        orig_device = next(model.parameters()).device
    except StopIteration:
        orig_device = torch.device('cpu')
    model = model.to(device)
    inp = torch.rand(input_size, device=device)
    flops, params = profile(model, inputs=(inp,), verbose=verbose) # verbose True for debugging
    if as_string:
        print(f"FLOPs: {flops/1e9:.2f} G, Params: {params/1e6:.2f} M")
    model = model.to(orig_device)
    return flops, params
