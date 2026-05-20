#!/usr/bin/env python3
"""
Export MisophoniaANCNet as a fixed-shape ONNX streaming step for Android.

Usage:
    python -m android.export \
        --checkpoint data/YOUR_RUN/checkpoints/best_weights.pt \
        --output export/misophonia_anc_step.onnx \
        --label-index 0 \
        --use-ema

Install export dependencies:
    pip install onnx onnxruntime

Android-side call pattern:
    x, enc_buf, dec_buf, out_buf = session.run(
        ["x", "new_enc_buf", "new_dec_buf", "new_out_buf"],
        {
            "mix": mix,
            "label": label,
            "enc_buf": enc_buf,
            "dec_buf": dec_buf,
            "out_buf": out_buf,
        },
    )
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from misophonia_anc.model import MisophoniaANCNet


class MobileANCStep(nn.Module):
    """
    Fixed-shape mobile inference wrapper.

    This wrapper exports one streaming step. It deliberately avoids:
      - dict inputs,
      - optional buffer initialization,
      - subtraction methods,
      - Python-side variable return types.

    Inputs:
        mix:     [1, 2, chunk_samples]
        label:   [1, label_len]
        enc_buf: encoder state buffer
        dec_buf: decoder state buffer
        out_buf: output-conv state buffer

    Outputs:
        x:           [1, 2, chunk_samples]
        new_enc_buf: updated encoder buffer
        new_dec_buf: updated decoder buffer
        new_out_buf: updated output buffer
    """

    def __init__(self, model: MisophoniaANCNet, chunk_samples: int) -> None:
        super().__init__()
        self.model = model.eval()
        self.chunk_samples = int(chunk_samples)

        # The exported mobile step should only output the direct model estimate.
        # Subtraction can be done outside the graph later if needed.
        self.model._subtraction_methods = {}

    def forward(
        self,
        mix: torch.Tensor,
        label: torch.Tensor,
        enc_buf: torch.Tensor,
        dec_buf: torch.Tensor,
        out_buf: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # For the current model, lookahead=True means the normal forward path
        # pads by L samples on both sides before predict().
        #
        # We hard-code the fixed-shape prototype case:
        #   chunk_samples must already be divisible by L.
        if self.model.lookahead:
            mix = F.pad(mix, (self.model.L, self.model.L))

        x, enc_buf, dec_buf, out_buf = self.model.predict(
            mix,
            label,
            enc_buf,
            dec_buf,
            out_buf,
        )

        return x, enc_buf, dec_buf, out_buf


def load_model_from_checkpoint(
    checkpoint_path: Path,
    *,
    use_ema: bool,
    device: torch.device,
) -> MisophoniaANCNet:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "hyperparameters" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain 'hyperparameters'. "
            "This script expects checkpoints saved with model.save_checkpoint()."
        )

    model_params = dict(checkpoint["hyperparameters"])
    model = MisophoniaANCNet(**model_params).to(device)

    if use_ema:
        ema_state = checkpoint.get("ema_model_state")
        if ema_state is None:
            raise KeyError("--use-ema was passed, but checkpoint does not contain 'ema_model_state'.")
        model.load_state_dict(ema_state)
    else:
        if "model_state" not in checkpoint:
            raise KeyError("Checkpoint does not contain 'model_state'.")
        model.load_state_dict(checkpoint["model_state"])

    model.eval()
    model._subtraction_methods = {}

    return model


def make_example_inputs(
    model: MisophoniaANCNet,
    *,
    label_index: int,
    chunk_samples: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    label_len = int(model.hyperparameters["label_len"])

    if not 0 <= label_index < label_len:
        raise ValueError(f"label_index must be in [0, {label_len - 1}], got {label_index}.")

    if chunk_samples % model.L != 0:
        raise ValueError(f"chunk_samples must be divisible by model.L={model.L}, got {chunk_samples}.")

    mix = torch.randn(1, 2, chunk_samples, dtype=torch.float32, device=device)

    label = torch.zeros(1, label_len, dtype=torch.float32, device=device)
    label[0, label_index] = 1.0

    enc_buf, dec_buf, out_buf = model.init_buffers(batch_size=1, device=device)

    return mix, label, enc_buf, dec_buf, out_buf


def export_onnx(
    wrapper: MobileANCStep,
    example_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        example_inputs,
        str(output_path),
        input_names=[
            "mix",
            "label",
            "enc_buf",
            "dec_buf",
            "out_buf",
        ],
        output_names=[
            "x",
            "new_enc_buf",
            "new_dec_buf",
            "new_out_buf",
        ],
        # Recommended modern exporter path; it is torch.export-based.
        # See PyTorch ONNX docs.
        dynamo=True,
        # Keep a single .onnx file if the model is below the ONNX 2GB limit.
        external_data=False,
        # Good default for current ONNX Runtime versions.
        opset_version=18,
        # Turn these on while debugging export failures.
        report=True,
        dump_exported_program=True,
        artifacts_dir=str(output_path.parent / "onnx_export_artifacts"),
    )


def verify_with_onnxruntime(
    onnx_path: Path,
    wrapper: MobileANCStep,
    example_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:

    wrapper.eval()

    # Clone because the PyTorch model mutates buffers in-place internally.
    pt_inputs = tuple(t.detach().cpu().clone() for t in example_inputs)

    with torch.inference_mode():
        pt_outputs = wrapper(*pt_inputs)

    ort_inputs = {
        "mix": example_inputs[0].detach().cpu().numpy().astype(np.float32),
        "label": example_inputs[1].detach().cpu().numpy().astype(np.float32),
        "enc_buf": example_inputs[2].detach().cpu().numpy().astype(np.float32),
        "dec_buf": example_inputs[3].detach().cpu().numpy().astype(np.float32),
        "out_buf": example_inputs[4].detach().cpu().numpy().astype(np.float32),
    }

    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )

    ort_outputs = session.run(
        ["x", "new_enc_buf", "new_dec_buf", "new_out_buf"],
        ort_inputs,
    )

    names = ["x", "new_enc_buf", "new_dec_buf", "new_out_buf"]
    for name, pt, ort_out in zip(names, pt_outputs, ort_outputs, strict=True):
        pt_np = pt.detach().cpu().numpy()
        max_abs_diff = float(np.max(np.abs(pt_np - ort_out)))
        mean_abs_diff = float(np.mean(np.abs(pt_np - ort_out)))
        print(f"{name}: max_abs_diff={max_abs_diff:.6g}, mean_abs_diff={mean_abs_diff:.6g}")


def save_mobile_metadata(
    output_path: Path,
    model: MisophoniaANCNet,
    example_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    label_index: int,
    chunk_samples: int,
    use_ema: bool,
) -> None:
    mix, label, enc_buf, dec_buf, out_buf = example_inputs

    metadata = {
        "onnx_model": str(output_path),
        "input_names": ["mix", "label", "enc_buf", "dec_buf", "out_buf"],
        "output_names": ["x", "new_enc_buf", "new_dec_buf", "new_out_buf"],
        "input_shapes": {
            "mix": list(mix.shape),
            "label": list(label.shape),
            "enc_buf": list(enc_buf.shape),
            "dec_buf": list(dec_buf.shape),
            "out_buf": list(out_buf.shape),
        },
        "output_audio_shape": [1, 2, chunk_samples],
        "chunk_samples": int(chunk_samples),
        "sample_rate": None,
        "label_index": int(label_index),
        "label_len": int(model.hyperparameters["label_len"]),
        "L": int(model.L),
        "model_dim": int(model.model_dim),
        "lookahead": bool(model.lookahead),
        "use_ema": bool(use_ema),
        "note": (
            "Android prototype should keep enc_buf, dec_buf, and out_buf between "
            "calls. If microphone input is mono, duplicate it to stereo before "
            "feeding 'mix'."
        ),
    }

    metadata_path = output_path.with_suffix(".mobile_metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"Wrote metadata: {metadata_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a checkpoint saved by MisophoniaANCNet.save_checkpoint().",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("export/misophonia_anc_step.onnx"),
        help="Path to write the ONNX model.",
    )
    parser.add_argument(
        "--label-index",
        type=int,
        required=True,
        help="Trigger-class index to set to 1 in the example one-hot label.",
    )
    parser.add_argument(
        "--chunk-samples",
        type=int,
        default=None,
        help=("Fixed audio samples per mobile inference step. Default: model.dec_chunk_size * model.L."),
    )
    parser.add_argument(
        "--use-ema",
        action="store_true",
        help="Export EMA weights instead of raw model weights.",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip ONNX Runtime numerical verification.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cpu")
    torch.set_grad_enabled(False)

    model = load_model_from_checkpoint(
        args.checkpoint,
        use_ema=args.use_ema,
        device=device,
    )

    # The cleanest prototype chunk is one decoder chunk in waveform samples.
    # With your defaults this is 72 * 8 = 576 samples.
    if args.chunk_samples is None:
        chunk_samples = int(model.mask_gen.decoder.chunk_size * model.L)
    else:
        chunk_samples = int(args.chunk_samples)

    wrapper = MobileANCStep(model, chunk_samples=chunk_samples).to(device).eval()

    example_inputs = make_example_inputs(
        model,
        label_index=args.label_index,
        chunk_samples=chunk_samples,
        device=device,
    )

    print("Export settings:")
    print(f"  checkpoint:     {args.checkpoint}")
    print(f"  output:         {args.output}")
    print(f"  use_ema:        {args.use_ema}")
    print(f"  chunk_samples:  {chunk_samples}")
    print(f"  label_index:    {args.label_index}")
    print(f"  label_len:      {model.hyperparameters['label_len']}")
    print(f"  L:              {model.L}")
    print(f"  dec_chunk_size: {model.mask_gen.decoder.chunk_size}")
    print(f"  lookahead:      {model.lookahead}")

    export_onnx(wrapper, example_inputs, args.output)
    print(f"Wrote ONNX model: {args.output}")

    save_mobile_metadata(
        args.output,
        model,
        example_inputs,
        label_index=args.label_index,
        chunk_samples=chunk_samples,
        use_ema=args.use_ema,
    )

    if not args.skip_verify:
        verify_with_onnxruntime(args.output, wrapper, example_inputs)


if __name__ == "__main__":
    main()
