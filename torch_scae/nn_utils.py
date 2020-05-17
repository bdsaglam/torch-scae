from typing import Tuple

import torch


def conv_output_size(in_size: int,
                     kernel_size: int,
                     stride: int = 1,
                     padding: int = 0) -> int:
    return (in_size - kernel_size + 2 * padding) // stride + 1


def conv_output_shape(input_shape: Tuple[int, int, int],
                      out_channels: int,
                      kernel_size: int,
                      stride: int = 1,
                      padding: int = 0) -> Tuple[int, int, int]:
    return (
        out_channels,
        conv_output_size(input_shape[1],
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding),
        conv_output_size(input_shape[2],
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding)
    )


def measure_shape(network, input_shape, input_dtype=torch.float32):
    with torch.no_grad():
        return network(torch.rand(1, *input_shape, dtype=input_dtype)).shape[1:]
