"""Scientific-only 16x16 -> 7x7 max pooling with fixed-order backward.

Forward values and argmax indices are delegated to PyTorch. Backward performs
49 explicitly ordered elementwise additions, without scatter/atomic reduction.
This is a first-order operator, not a replacement for general adaptive pooling.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


OPERATOR_ID = "scientific-adaptive-maxpool-16x16-to-7x7-fixed-row-major-v1"


def operator_identity():
    """Return an independent JSON-compatible declaration (no state or RNG)."""
    return {
        "operator_id": OPERATOR_ID,
        "input_layout": "strided NCHW; N>=1,C>=1; noncontiguous supported",
        "input_spatial_shape": [16, 16],
        "output_spatial_shape": [7, 7],
        "dtypes": ["torch.float32", "torch.float64"],
        "devices": ["cpu", "cuda"],
        "forward": "native torch.nn.functional.adaptive_max_pool2d with returned indices",
        "backward": "saved-index routing; 49 output bins in row-major order; out-of-place addition",
        "autograd_order": 1,
        "nonfinite_policy": "no new filtering; where routes each upstream value only to its saved index",
    }


def _validate_input(value):
    if not isinstance(value, torch.Tensor):
        raise TypeError("Scientific pooling requires a torch.Tensor")
    if value.layout != torch.strided:
        raise TypeError("Scientific pooling supports only strided tensors")
    if value.dtype not in (torch.float32, torch.float64):
        raise TypeError("Scientific pooling supports only float32 and float64")
    if value.device.type not in ("cpu", "cuda"):
        raise ValueError("Scientific pooling supports CPU/CUDA only; no device fallback")
    if value.ndim != 4 or value.shape[0] < 1 or value.shape[1] < 1 or tuple(value.shape[-2:]) != (16, 16):
        raise ValueError("Scientific pooling requires NCHW with N>=1,C>=1 and spatial shape16x16")


class _FixedOrderAdaptiveMaxPool2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        _validate_input(value)
        output, indices = F.adaptive_max_pool2d(value, (7, 7), return_indices=True)
        ctx.save_for_backward(indices)
        ctx.input_shape = tuple(value.shape)
        ctx.input_dtype = value.dtype
        ctx.input_device = value.device
        ctx.mark_non_differentiable(indices)
        ctx.set_materialize_grads(False)
        return output, indices

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        if torch.is_grad_enabled():
            raise RuntimeError("Scientific fixed-order pooling supports first-order autograd only; create_graph is unsupported")
        if grad_indices is not None:
            raise RuntimeError("Scientific pooling indices are nondifferentiable")
        if grad_output is None:
            return None
        if grad_output.dtype != ctx.input_dtype or grad_output.device != ctx.input_device:
            raise RuntimeError("Scientific pooling backward dtype/device differs from forward")
        if tuple(grad_output.shape) != (*ctx.input_shape[:2], 7, 7):
            raise RuntimeError("Scientific pooling backward shape differs from forward")
        (indices,) = ctx.saved_tensors
        n, channels = ctx.input_shape[:2]
        saved = indices.reshape(n, channels, 49)
        upstream = grad_output.reshape(n, channels, 49)
        spatial = torch.arange(256, dtype=torch.int64, device=ctx.input_device).reshape(1, 1, 256)
        accumulated = torch.zeros((n, channels, 256), dtype=ctx.input_dtype, device=ctx.input_device)
        zero = torch.zeros((), dtype=ctx.input_dtype, device=ctx.input_device)
        for output_bin in range(49):
            selected = saved[:, :, output_bin:output_bin + 1] == spatial
            # where, rather than mask * gradient, avoids spreading 0 * NaN.
            contribution = torch.where(selected, upstream[:, :, output_bin:output_bin + 1], zero)
            accumulated = accumulated + contribution
        return accumulated.reshape(ctx.input_shape)


def adaptive_max_pool2d_fixed_order(value, return_indices=False):
    """Native maxima/ties, same-device fixed-order first derivative.

    No determinism setting is changed. CUDA backward of native adaptive max
    pooling is never called. Higher-order autograd is explicitly rejected.
    """
    if not isinstance(return_indices, bool):
        raise TypeError("return_indices must be a bool")
    output, indices = _FixedOrderAdaptiveMaxPool2d.apply(value)
    return (output, indices) if return_indices else output
