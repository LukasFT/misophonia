# ruff: noqa: N803, N806

from typing import Callable, TypeAlias

import numpy as np
import torch

SubtractionMethod: TypeAlias = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
"""A subtraction method is: (mix, pred) -> output, where mix and pred are stereo tensors of shape (2, T) and output is the same shape."""


def simple_subtraction(mix: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """
    Remove the trigger audio from the mix audio using simple subtraction.
    """
    # assert audio_mix.shape[0] == 2, "Audio mix must be stereo (2 channels)"
    # assert audio_mix.shape == audio_trigger.shape, "Audio signals must have the same shape"
    return mix - pred


def stft_subtraction(mix: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """
    Remove the trigger audio from the mix audio using STFT.

    Produces a soft (Wiener-like) time-frequency mask and applies it to the mixture STFT
    (preserving mixture phase). Uses sqrt-Hann + overlap-add normalization for stable gain.
    Finally rescales output peak to match input peak per channel (prevents huge ranges).
    """
    # TODO: Change implementation to use torch
    # TODO: Test this method - might be incorrect

    # ---- STFT/ISTFT helpers (NumPy-only) ----
    def _hann(N: int) -> np.ndarray:
        n = np.arange(N, dtype=np.float64)
        return 0.5 - 0.5 * np.cos(2.0 * np.pi * n / N)

    def _stft_1ch(x: np.ndarray, n_fft: int, hop: int, win: np.ndarray) -> tuple[np.ndarray, int]:
        x = x.astype(np.float64, copy=False)
        T = x.shape[0]
        n_frames = 1 if T <= n_fft else 1 + int(np.ceil((T - n_fft) / hop))
        pad = (n_frames - 1) * hop + n_fft - T
        if pad > 0:
            x = np.pad(x, (0, pad), mode="constant")
        frames = np.lib.stride_tricks.as_strided(
            x,
            shape=(n_frames, n_fft),
            strides=(x.strides[0] * hop, x.strides[0]),
            writeable=False,
        )
        frames_win = frames * win[None, :]
        X = np.fft.rfft(frames_win, n=n_fft, axis=-1)
        return X, T

    def _istft_1ch(X: np.ndarray, n_fft: int, hop: int, win: np.ndarray, length: int) -> np.ndarray:
        n_frames = X.shape[0]
        y_len = (n_frames - 1) * hop + n_fft
        y = np.zeros(y_len, dtype=np.float64)
        wsum = np.zeros(y_len, dtype=np.float64)

        for i in range(n_frames):
            frame = np.fft.irfft(X[i], n=n_fft)
            start = i * hop
            y[start : start + n_fft] += frame * win
            wsum[start : start + n_fft] += win * win  # window power for perfect reconstruction normalization

        nonzero = wsum > 1e-12
        y[nonzero] /= wsum[nonzero]
        return y[:length]

    # ---- Parameters ----
    n_fft = 1024
    hop = n_fft // 4
    # sqrt-Hann is a robust analysis/synthesis choice with OLA normalization
    win = np.sqrt(_hann(n_fft))

    eps = 1e-10
    power = 2.0  # Wiener-like
    # How strongly to suppress trigger-dominant bins; 1.0 = normal, >1.0 = more aggressive
    mask_sharpness = 1.0

    mix = mix.astype(np.float64, copy=False)
    trig = pred.astype(np.float64, copy=False)

    out = np.zeros_like(mix, dtype=np.float64)

    for ch in range(2):
        X, Tlen = _stft_1ch(mix[ch], n_fft=n_fft, hop=hop, win=win)
        T_est, _ = _stft_1ch(trig[ch], n_fft=n_fft, hop=hop, win=win)

        magX = np.abs(X)
        magT = np.abs(T_est)

        # Background magnitude proxy: clip(mix - trigger, 0)
        magB = np.maximum(magX - magT, 0.0)

        Bp = magB**power
        Tp = magT**power

        M = Bp / (Bp + Tp + eps)  # in [0,1]
        if mask_sharpness != 1.0:
            M = M**mask_sharpness  # optional shaping

        Y = X * M
        out[ch] = _istft_1ch(Y, n_fft=n_fft, hop=hop, win=win, length=Tlen)

        # ---- Peak match to mixture per channel (prevents the huge dynamic-range blowup you saw) ----
        in_peak = np.max(np.abs(mix[ch])) + eps
        out_peak = np.max(np.abs(out[ch])) + eps
        out[ch] *= in_peak / out_peak

    return out.astype(mix.dtype, copy=False)


def ls_fir_subtraction(mix: np.ndarray, pred: np.ndarray, filter_len: int = 33, ridge: float = 1e-4) -> np.ndarray:
    """
    Remove trigger via a least-squares FIR matching filter per channel.

    Fits a short FIR h such that conv(trigger, h) matches the trigger contribution inside the mixture,
    then subtracts that reconstructed contribution.

    Args:
        audio_mix (np.ndarray): Stereo mixture, shape (2, T).
        audio_trigger (np.ndarray): Stereo trigger estimate, shape (2, T).
        filter_len (int): FIR length (typical 17-65). Larger handles more mismatch but risks overfitting.
        ridge (float): L2 regularization for stability.

    """
    assert mix.shape[0] == 2, "Audio mix must be stereo (2 channels)"
    assert mix.shape == pred.shape, "Audio signals must have the same shape"

    x = mix.astype(np.float64, copy=False)
    t = pred.astype(np.float64, copy=False)

    eps = 1e-12
    out = np.zeros_like(x, dtype=np.float64)

    def _convmtx_valid(sig: np.ndarray, L: int) -> np.ndarray:
        """
        Build a convolution matrix X so that X @ h = valid_conv(sig, h),
        i.e., length (T-L+1), with standard convolution tap order.
        """
        T = sig.shape[0]
        if L > T:
            raise ValueError("filter_len must be <= number of samples")
        shape = (T - L + 1, L)
        strides = (sig.strides[0], sig.strides[0])
        X = np.lib.stride_tricks.as_strided(sig, shape=shape, strides=strides, writeable=False)
        return X[:, ::-1].copy()  # reverse so dot matches convolution

    for ch in range(2):
        x1 = x[ch]
        t1 = t[ch]
        Tlen = x1.shape[0]
        L = int(min(filter_len, Tlen))
        if L < 1:
            out[ch] = x1
            continue

        X = _convmtx_valid(t1, L)  # (T-L+1, L)
        y = x1[L - 1 :]  # align to valid conv output

        # Solve (X^T X + ridge I) h = X^T y
        XtX = X.T @ X
        XtX.flat[:: XtX.shape[0] + 1] += ridge + eps
        Xty = X.T @ y
        h = np.linalg.solve(XtX, Xty)

        trigger_fit_valid = X @ h  # (T-L+1,)

        bg = x1.copy()
        bg[L - 1 :] = y - trigger_fit_valid
        out[ch] = bg

    return out.astype(mix.dtype, copy=False)
