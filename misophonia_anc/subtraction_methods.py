# ruff: noqa: N803, N806

from typing import Callable, TypeAlias

import torch

SubtractionMethod: TypeAlias = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
"""A subtraction method is: (mix, pred) -> output, where mix and pred are stereo tensors of shape (2, T) and output is the same shape."""


def simple_subtraction(mix: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """
    Remove the trigger audio from the mix audio using simple subtraction.
    """
    # assert audio_mix.shape[0] == 2, "Audio mix must be stereo (2 channels)"
    # assert audio_mix.shape == audio_trigger.shape, "Audio signals must have the same shape"
    # mix, pred: (B, 2, T)
    eps = torch.finfo(mix.dtype).tiny
    max_gain = 1.5

    alpha = (mix * pred).sum(dim=-1, keepdim=True) / (
        pred.pow(2).sum(dim=-1, keepdim=True) + eps
    )
    alpha = alpha.clamp(0.0, max_gain)

    return mix - pred * alpha


def stft_subtraction(mix: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """
    Remove trigger audio from mix using STFT masking.

    Args:
        mix: Tensor, shape (B, 2, T).
        pred: Tensor, shape (B, 2, T).

    Returns:
        Tensor, shape (B, 2, T).
    """
    assert mix.ndim == 3, "mix must have shape (B, 2, T)"
    assert mix.shape[1] == 2, "Audio mix must be stereo (2 channels)"
    assert mix.shape == pred.shape, "Audio signals must have the same shape"

    device = mix.device
    dtype = mix.dtype

    n_fft = 1024
    hop = n_fft // 4
    eps = 1e-10
    power = 2.0
    mask_sharpness = 1.0

    win = torch.sqrt(torch.hann_window(n_fft, periodic=True, device=device, dtype=dtype))

    out_channels = []

    for ch in range(2):
        x = mix[:, ch, :]  # (B, T)
        t = pred[:, ch, :]  # (B, T)
        T = x.shape[-1]

        X = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=win,
            center=True,
            return_complex=True,
        )

        T_est = torch.stft(
            t,
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=win,
            center=True,
            return_complex=True,
        )

        magX = torch.abs(X)
        magT = torch.abs(T_est)

        magB = torch.clamp(magX - magT, min=0.0)

        Bp = magB.pow(power)
        Tp = magT.pow(power)

        M = Bp / (Bp + Tp + eps)

        if mask_sharpness != 1.0:
            M = M.pow(mask_sharpness)

        Y = X * M

        y = torch.istft(
            Y,
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=win,
            center=True,
            length=T,
        )  # (B, T)

        # Peak-match per example, per channel. Uncomment to enable.
        # in_peak = torch.amax(torch.abs(x), dim=-1, keepdim=True) + eps
        # out_peak = torch.amax(torch.abs(y), dim=-1, keepdim=True) + eps
        # y = y * (in_peak / out_peak)

        out_channels.append(y)

    return torch.stack(out_channels, dim=1).to(device=device, dtype=dtype)


def ls_fir_subtraction(
    mix: torch.Tensor,
    pred: torch.Tensor,
    filter_len: int = 33,
    ridge: float = 1e-4,
) -> torch.Tensor:
    """
    Remove trigger via batched least-squares FIR matching per channel.

    Args:
        mix: Tensor, shape (B, 2, T).
        pred: Tensor, shape (B, 2, T).
        filter_len: FIR length.
        ridge: L2 regularization strength.

    Returns:
        Tensor, shape (B, 2, T).
    """
    assert mix.ndim == 3, "mix must have shape (B, 2, T)"
    assert mix.shape[1] == 2, "Audio mix must be stereo (2 channels)"
    assert mix.shape == pred.shape, "Audio signals must have the same shape"

    device = mix.device
    dtype = mix.dtype

    x = mix.to(torch.float64)
    t = pred.to(torch.float64)

    B, C, T = x.shape
    eps = 1e-12
    out = torch.zeros_like(x)

    def _convmtx_valid(sig: torch.Tensor, L: int) -> torch.Tensor:
        """
        Build batched convolution matrix.

        Args:
            sig: Tensor, shape (B, T).
            L: FIR length.

        Returns:
            Tensor, shape (B, T - L + 1, L).
        """
        if L > sig.shape[-1]:
            raise ValueError("filter_len must be <= number of samples")

        X = sig.unfold(dimension=-1, size=L, step=1)
        return torch.flip(X, dims=[-1]).contiguous()

    L = int(min(filter_len, T))

    if L < 1:
        return mix.clone()

    eye = torch.eye(L, device=device, dtype=torch.float64)

    for ch in range(2):
        x1 = x[:, ch, :]  # (B, T)
        t1 = t[:, ch, :]  # (B, T)

        X = _convmtx_valid(t1, L)  # (B, T - L + 1, L)
        y = x1[:, L - 1 :]  # (B, T - L + 1)

        Xt = X.transpose(-1, -2)  # (B, L, T - L + 1)

        XtX = Xt @ X  # (B, L, L)
        XtX = XtX + (ridge + eps) * eye.unsqueeze(0)

        Xty = Xt @ y.unsqueeze(-1)  # (B, L, 1)

        h = torch.linalg.solve(XtX, Xty)  # (B, L, 1)

        trigger_fit_valid = (X @ h).squeeze(-1)  # (B, T - L + 1)

        bg = x1.clone()
        bg[:, L - 1 :] = y - trigger_fit_valid

        out[:, ch, :] = bg

    return out.to(device=device, dtype=dtype)
