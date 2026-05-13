"""
Encoder logic.

Heavily based on https://github.com/vb000/SemanticHearing
"""

# ruff: noqa: ANN001, ANN002, ANN003 # TODO: Improve quality

from collections import OrderedDict

import torch
import torch.nn as nn
from torch import Tensor


class DilatedCausalConvEncoder(nn.Module):
    """
    A dilated causal convolution based encoder for encoding
    time domain audio input into latent space.
    """

    def __init__(
        self,
        *,
        channels,
        num_layers,
        dropout: float | None = None,  # None for backwards compatibility (equivalent to 0.0)
        kernel_size=3,
    ) -> None:
        super(DilatedCausalConvEncoder, self).__init__()
        self.channels = channels
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.dropout = nn.Dropout(p=dropout) if dropout is not None else None

        # Compute buffer lengths for each layer
        # buf_length[i] = (kernel_size - 1) * dilation[i]
        self.buf_lengths = [(kernel_size - 1) * 2**i for i in range(num_layers)]

        # Compute buffer start indices for each layer
        self.buf_indices = [0]
        for i in range(num_layers - 1):
            self.buf_indices.append(self.buf_indices[-1] + self.buf_lengths[i])

        # Dilated causal conv layers aggregate previous context to obtain
        # contexful encoded input.
        _dcc_layers = OrderedDict()
        for i in range(num_layers):
            dcc_layer = _DepthwiseSeparableConv(channels, channels, kernel_size=3, stride=1, padding=0, dilation=2**i)
            _dcc_layers.update({"dcc_%d" % i: dcc_layer})
        self.dcc_layers = nn.Sequential(_dcc_layers)

    def init_ctx_buf(self, batch_size, device) -> Tensor:
        """
        Returns an initialized context buffer for a given batch size.
        """
        return torch.zeros(
            (batch_size, self.channels, (self.kernel_size - 1) * (2**self.num_layers - 1)), device=device
        )

    def forward(self, x, ctx_buf) -> Tensor:
        """
        Encodes input audio `x` into latent space, and aggregates
        contextual information in `ctx_buf`. Also generates new context
        buffer with updated context.
        Args:
            x: [B, in_channels, T]
                Input multi-channel audio.
            ctx_buf: {[B, channels, self.buf_length[0]], ...}
                A list of tensors holding context for each dilation
                causal conv layer. (len(ctx_buf) == self.num_layers)
        Returns:
            ctx_buf: {[B, channels, self.buf_length[0]], ...}
                Updated context buffer with output as the
                last element.
        """
        # Note Sequence length T = x.shape[-1]

        for i in range(self.num_layers):
            buf_start_idx = self.buf_indices[i]
            buf_end_idx = self.buf_indices[i] + self.buf_lengths[i]

            # DCC input: concatenation of current output and context
            dcc_in = torch.cat((ctx_buf[..., buf_start_idx:buf_end_idx], x), dim=-1)

            # Push current output to the context buffer
            ctx_buf[..., buf_start_idx:buf_end_idx] = dcc_in[..., -self.buf_lengths[i] :]

            # Residual connection
            res = self.dcc_layers[i](dcc_in)
            if self.dropout is not None:
                res = self.dropout(res)
            x = x + res

        return x, ctx_buf


class _LayerNormPermuted(nn.LayerNorm):
    def __init__(self, *args, **kwargs) -> None:
        super(_LayerNormPermuted, self).__init__(*args, **kwargs)

    def forward(self, x):  # noqa: ANN202
        """
        Args:
            x: [B, C, T]
        """
        x = x.permute(0, 2, 1)  # [B, T, C]
        x = super().forward(x)
        x = x.permute(0, 2, 1)  # [B, C, T]
        return x


class _DepthwiseSeparableConv(nn.Module):
    """
    Depthwise separable convolutions
    """

    def __init__(self, in_channels, out_channels, *, kernel_size, stride, padding, dilation) -> None:
        super(_DepthwiseSeparableConv, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, dilation=dilation),
            _LayerNormPermuted(in_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            _LayerNormPermuted(out_channels),
            nn.ReLU(),
        )

    def forward(self, x) -> Tensor:
        return self.layers(x)
