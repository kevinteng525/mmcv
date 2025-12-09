# Copyright 2019 Yan Yan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TorchScript-compatible implementations of sparse operations.

This module provides TorchScript-compatible replacements for the external
C++/CUDA extensions, enabling model export without requiring users to
modify their model definitions.
"""

import torch
from typing import Tuple, Optional
import os


# Check if we're in TorchScript export mode
# This allows us to use the original fast implementation during training
# and fall back to TS-compatible implementation during export
_torchscript_export_mode = False
_original_ext_module = None


def set_torchscript_export_mode(enabled: bool):
    """Enable or disable TorchScript export mode."""
    global _torchscript_export_mode
    _torchscript_export_mode = enabled


def get_torchscript_export_mode() -> bool:
    """Check if TorchScript export mode is enabled."""
    return _torchscript_export_mode

@torch.jit.script
def _indice_conv_forward_impl(features: torch.Tensor,
                              filters: torch.Tensor,
                              indice_pairs: torch.Tensor,
                              indice_pair_num: torch.Tensor,
                              num_activate_out: int,
                              inverse: bool = False,
                              subm: bool = False) -> torch.Tensor:
    device = features.device
    num_in_feats = features.shape[0]
    in_channels = features.shape[1]
    out_channels = filters.shape[-1]

    # Determine kernel dimension
    ndim = len(filters.shape) - 2
    kernel_size = filters.shape[:ndim]
    num_kernels = int(torch.prod(torch.tensor(kernel_size)).item())

    # Initialize output
    output = torch.zeros(num_activate_out, out_channels, device=device, dtype=features.dtype)

    # Reshape filters
    filters_reshaped = filters.view(-1, in_channels, out_channels)

    # 修正：累积索引而不是成对读取
    pair_offset = 0

    for k in range(num_kernels):
        # 方案A：如果 indice_pair_num 存储的是每个kernel的pair数量
        num_pairs = int(indice_pair_num[k].item())

        if num_pairs == 0:
            continue

        pair_start = pair_offset
        pair_end = pair_offset + num_pairs
        pair_offset = pair_end  # 更新偏移量

        # 后续逻辑保持不变
        active_pairs = indice_pairs[pair_start:pair_end, :]
        if active_pairs.numel() == 0:
            continue

        in_indices = active_pairs[:, 0].long()
        out_indices = active_pairs[:, 1].long()

        in_mask = (in_indices >= 0) & (in_indices < num_in_feats)
        out_mask = (out_indices >= 0) & (out_indices < num_activate_out)
        valid_mask = in_mask & out_mask

        if not valid_mask.any():
            continue

        valid_in_indices = in_indices[valid_mask]
        valid_out_indices = out_indices[valid_mask]

        active_input_features = features[valid_in_indices]
        kernel_weight = filters_reshaped[k]
        conv_output = torch.mm(active_input_features, kernel_weight)

        if subm:
            output.index_add(0, valid_out_indices, conv_output)
        elif inverse:
            output.index_add(0, valid_in_indices, conv_output)
        else:
            output.index_add(0, valid_out_indices, conv_output)

    return output


@torch.jit.script
def _indice_conv_backward_impl(features: torch.Tensor,
                               filters: torch.Tensor,
                               out_bp: torch.Tensor,
                               indice_pairs: torch.Tensor,
                               indice_pair_num: torch.Tensor,
                               inverse: bool = False,
                               subm: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    TorchScript-compatible implementation of indice_conv_backward.

    Args:
        features: Input features from forward pass [N, C_in]
        filters: Convolution filters [K1, K2, K3, C_in, C_out]
        out_bp: Gradient of output [M, C_out]
        indice_pairs: Pairs of indices from forward pass
        indice_pair_num: Number of pairs per kernel position
        inverse: Whether this was an inverse convolution
        subm: Whether this was a submanifold convolution

    Returns:
        Tuple of (grad_input, grad_filters)
    """
    device = features.device
    num_in_feats = features.shape[0]
    in_channels = features.shape[1]
    num_out_feats = out_bp.shape[0]
    out_channels = out_bp.shape[1]

    # Determine kernel dimension
    ndim = len(filters.shape) - 2
    kernel_size = filters.shape[:ndim]
    num_kernels = int(torch.prod(torch.tensor(kernel_size)).item())

    # Initialize gradients
    grad_input = torch.zeros_like(features)
    grad_filters = torch.zeros_like(filters)

    # Reshape filters for easier indexing [num_kernels, in_channels, out_channels]
    filters_reshaped = filters.view(-1, in_channels, out_channels)
    grad_filters_reshaped = grad_filters.view(-1, in_channels, out_channels)

    # Process each kernel position
    for k in range(num_kernels):
        # Get indices for this kernel position
        pair_start = int(indice_pair_num[2 * k].item())
        pair_end = int(indice_pair_num[2 * k + 1].item())

        if pair_start >= pair_end:
            continue

        # Get active pairs for this kernel position
        active_pairs = indice_pairs[pair_start:pair_end, :]
        if active_pairs.numel() == 0:
            continue

        # Extract input and output indices
        in_indices = active_pairs[:, 0].long()
        out_indices = active_pairs[:, 1].long()

        # Bounds checking
        in_mask = (in_indices >= 0) & (in_indices < num_in_feats)
        out_mask = (out_indices >= 0) & (out_indices < num_out_feats)
        valid_mask = in_mask & out_mask

        if not valid_mask.any():
            continue

        # Use only valid indices
        valid_in_indices = in_indices[valid_mask]
        valid_out_indices = out_indices[valid_mask]

        # Get corresponding gradients
        active_grad_output = out_bp[valid_out_indices]  # [P, C_out]
        active_input_features = features[valid_in_indices]  # [P, C_in]

        # Compute gradient for filters: grad = input.T @ grad_output
        # [C_in, P] x [P, C_out] -> [C_in, C_out]
        grad_filters_k = torch.mm(active_input_features.t(), active_grad_output)
        grad_filters_reshaped[k] = grad_filters_k

        # Compute gradient for input: grad = grad_output @ filter.T
        # [P, C_out] x [C_out, C_in] -> [P, C_in]
        kernel_weight = filters_reshaped[k]  # [C_in, C_out]
        grad_input_feature = torch.mm(active_grad_output, kernel_weight.t())  # [P, C_in]

        # Accumulate gradient for input
        if subm:
            # Submanifold: direct assignment
            grad_input.index_add(0, valid_in_indices, grad_input_feature)
        elif inverse:
            # Inverse convolution
            grad_input.index_add(0, valid_out_indices, grad_input_feature)
        else:
            # Standard convolution
            grad_input.index_add(0, valid_in_indices, grad_input_feature)

    return grad_input, grad_filters

@torch.jit.script
def _indice_maxpool_forward_impl(features: torch.Tensor,
                                 indice_pairs: torch.Tensor,
                                 indice_pair_num: torch.Tensor,
                                 num_activate_out: int) -> torch.Tensor:
    """
    TorchScript-compatible implementation of indice_maxpool_forward.
    """
    device = features.device
    channels = features.shape[1]

    # Initialize output with very small values
    output = torch.full((num_activate_out, channels),
                       float('-inf'),
                       device=device,
                       dtype=features.dtype)

    # Process each pooling region
    num_regions = indice_pair_num.shape[0] // 2

    for r in range(num_regions):
        # Get indices for this pooling region
        pair_start = int(indice_pair_num[2 * r].item())
        pair_end = int(indice_pair_num[2 * r + 1].item())

        if pair_start >= pair_end:
            continue

        # Get active indices for this region
        active_pairs = indice_pairs[pair_start:pair_end, :]
        if active_pairs.numel() == 0:
            continue

        in_indices = active_pairs[:, 0].long()
        out_indices = active_pairs[:, 1].long()

        # Bounds checking
        in_mask = (in_indices >= 0) & (in_indices < features.shape[0])
        out_mask = (out_indices >= 0) & (out_indices < num_activate_out)
        valid_mask = in_mask & out_mask

        if not valid_mask.any():
            continue

        valid_in_indices = in_indices[valid_mask]
        valid_out_indices = out_indices[valid_mask]

        # For each output position, take max over corresponding inputs
        unique_out = torch.unique(valid_out_indices)
        for out_idx in unique_out:
            mask = valid_out_indices == out_idx
            if mask.any():
                input_indices = valid_in_indices[mask]
                pool_features = features[input_indices]  # [P, C]
                max_values, _ = torch.max(pool_features, dim=0)
                output[out_idx] = torch.max(output[out_idx], max_values)

    return output

@torch.jit.script
def _indice_maxpool_backward_impl(features: torch.Tensor,
                                  out_features: torch.Tensor,
                                  out_bp: torch.Tensor,
                                  indice_pairs: torch.Tensor,
                                  indice_pair_num: torch.Tensor) -> torch.Tensor:
    """
    TorchScript-compatible implementation of indice_maxpool_backward.
    """
    device = features.device
    grad_input = torch.zeros_like(features)

    # Process each pooling region
    num_regions = indice_pair_num.shape[0] // 2

    for r in range(num_regions):
        # Get indices for this pooling region
        pair_start = int(indice_pair_num[2 * r].item())
        pair_end = int(indice_pair_num[2 * r + 1].item())

        if pair_start >= pair_end:
            continue

        # Get active indices for this region
        active_pairs = indice_pairs[pair_start:pair_end, :]
        if active_pairs.numel() == 0:
            continue

        in_indices = active_pairs[:, 0].long()
        out_indices = active_pairs[:, 1].long()

        # Bounds checking
        in_mask = (in_indices >= 0) & (in_indices < features.shape[0])
        out_mask = (out_indices >= 0) & (out_indices < out_features.shape[0])
        valid_mask = in_mask & out_mask

        if not valid_mask.any():
            continue

        valid_in_indices = in_indices[valid_mask]
        valid_out_indices = out_indices[valid_mask]

        # For each output position, distribute gradient to max elements
        unique_out = torch.unique(valid_out_indices)
        for out_idx in unique_out:
            mask = valid_out_indices == out_idx
            if mask.any():
                input_indices = valid_in_indices[mask]

                # Get output gradient
                out_grad = out_bp[out_idx]  # [C]

                # Find which inputs had the max value
                pool_features = features[input_indices]  # [P, C]
                max_features = out_features[out_idx]  # [C]

                # Create mask for max elements
                is_max = (pool_features == max_features.unsqueeze(0))  # [P, C]

                # Distribute gradient to max elements
                # If multiple elements have the same max, average the gradient
                max_count = torch.sum(is_max, dim=0, keepdim=True)  # [1, C]
                max_count = torch.clamp(max_count, min=1)  # Avoid division by zero

                grad_per_max = out_grad.unsqueeze(0) / max_count  # [P, C]
                grad_input[input_indices] += grad_per_max * is_max.float()

    return grad_input


def indice_conv_forward(features, filters, indice_pairs, indice_pair_num,
                       num_activate_out, inverse=False, subm=False):
    """
    Wrapper for indice_conv_forward that automatically uses TS implementation
    when in TorchScript export mode.
    """
    # Use TorchScript-compatible implementation
    return _indice_conv_forward_impl(
        features, filters, indice_pairs, indice_pair_num,
        num_activate_out, inverse, subm)


def indice_conv_backward(features, filters, out_bp, indice_pairs,
                        indice_pair_num, inverse=False, subm=False):
    """
    Wrapper for indice_conv_backward that automatically uses TS implementation
    when in TorchScript export mode.
    """
    return _indice_conv_backward_impl(
        features, filters, out_bp, indice_pairs,
        indice_pair_num, inverse, subm)


def indice_maxpool_forward(features, indice_pairs, indice_pair_num,
                          num_activate_out):
    """
    Wrapper for indice_maxpool_forward that automatically uses TS implementation
    when in TorchScript export mode.
    """
    return _indice_maxpool_forward_impl(
        features, indice_pairs, indice_pair_num, num_activate_out)


def indice_maxpool_backward(features, out_features, out_bp,
                           indice_pairs, indice_pair_num):
    """
    Wrapper for indice_maxpool_backward that automatically uses TS implementation
    when in TorchScript export mode.
    """
        # Use TorchScript-compatible implementation
    return _indice_maxpool_backward_impl(
        features, out_features, out_bp, indice_pairs, indice_pair_num)