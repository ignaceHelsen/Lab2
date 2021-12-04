import torch

import utilities.utils as utils


def mse(input_tensor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """error = input_tensor - target
    return error.pow(2).mean()"""
    # OR
    error = torch.square(input_tensor - target)
    return error.mean()

