# ruff: noqa: ANN001, ANN002, ANN003 # TODO: Improve quality


# TODO: Remove unused (commented out) functions
# import os
# from copy import deepcopy


import sys
from pathlib import Path


import numpy as np
import torch
import torch.nn.functional as F
# import torch.nn as nn
# import torch.optim as optim
# import torchaudio
# from torchmetrics.functional import (
#     scale_invariant_signal_noise_ratio as si_snr,
# )
# from torchmetrics.functional import (
#     signal_noise_ratio as snr,
# )

# from ._utils import ild_diff, itd_diff, mod_pad
# from .mask_net import MaskNet


from misophonia_anc._utils import mod_pad
from misophonia_dataset.interface import MisophoniaItem


def custom_collate_fn(
    batch: list[list[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    # Pad the audio to all be the same length (the length of the longest audio in the batch)
    max_len = max([mix.shape[-1] for mix, _, _ in batch])

    mixes = []
    gts = []
    labels = []
    masks = []
    for mix, label, gt in batch:
        pad_len = max_len - mix.shape[-1]
        assert pad_len >= 0, "Error calculating batch padding"

        mix = F.pad(torch.from_numpy(mix).to(torch.float32), (0, pad_len))  # Convert and pad mix
        gt = F.pad(torch.from_numpy(gt).to(torch.float32), (0, pad_len))  # Convert and pad gt

        mask = torch.zeros_like(mix)
        mask[:, :, -pad_len:] = 1.0

        mixes.append(mix)
        gts.append(gt)
        labels.append(torch.from_numpy(label).to(torch.float32))  # Convert label
        masks.append(mask)

    inputs = {
        "mix": torch.stack(mixes),
        "label_vector": torch.stack(labels),
    }
    gt = torch.stack(gts)
    masks = torch.stack(masks)

    return inputs, gt, masks

# def optimizer(  # noqa: ANN201
#     model,
#     data_parallel=False,  # TODO: Remove unused parameter?  # noqa: FBT002
#     **kwargs,
# ):
#     params = [p for p in model.parameters() if p.requires_grad]
#     return optim.Adam(params, **kwargs)


# def loss(_output, tgt) -> torch.Tensor:
#     pred = _output["x"]
#     return -0.9 * snr(pred, tgt).mean() - 0.1 * si_snr(pred, tgt).mean()


# def metrics(inputs, _output, gt) -> dict:
#     """Function to compute metrics"""
#     mixed = inputs["mixture"]
#     output = _output["x"]
#     metrics = {}

#     def _metric_i(metric, src, pred, tgt):  # noqa: ANN202
#         _vals = []
#         for s, t, p in zip(src, tgt, pred):
#             _vals.append(torch.mean((metric(p, t) - metric(s, t))).cpu().item())
#         return _vals

#     for m_fn in [snr, si_snr]:
#         metrics[m_fn.__name__] = _metric_i(m_fn, mixed[:, : gt.shape[1], :], output, gt)

#     return metrics


# def test_metrics(inputs, _output, gt) -> dict:
#     test_metrics = metrics(inputs, _output, gt)
#     output = _output["x"]
#     delta_itds, delta_ilds, snrs = [], [], []
#     for o, g in zip(output, gt):
#         delta_itds.append(itd_diff(o.cpu(), g.cpu(), sr=44100))
#         delta_ilds.append(ild_diff(o.cpu().numpy(), g.cpu().numpy()))
#         snrs.append(torch.mean(si_snr(o, g)).cpu().item())
#     test_metrics["delta_ITD"] = delta_itds
#     test_metrics["delta_ILD"] = delta_ilds
#     test_metrics["si_snr"] = snrs
#     return test_metrics


# def format_results(idx, inputs, output, gt, metrics, output_dir=None) -> dict:
#     results = metrics
#     results["metadata"] = inputs["metadata"]
#     results = deepcopy(results)

#     # Save audio
#     if output_dir is not None:
#         output = output["x"]
#         for i in range(output.shape[0]):
#             out_dir = os.path.join(output_dir, f"{idx + i:03d}")
#             os.makedirs(out_dir)
#             torchaudio.save(os.path.join(out_dir, "mixture.wav"), inputs["mixture"][i], 44100)
#             torchaudio.save(os.path.join(out_dir, "gt.wav"), gt[i], 44100)
#             torchaudio.save(os.path.join(out_dir, "output.wav"), output[i], 44100)

#     return results
