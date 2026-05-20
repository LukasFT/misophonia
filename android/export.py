#!/usr/bin/env python3
"""
Export one or more MisophoniaANCNet checkpoints as fixed-shape ONNX streaming
steps for the Android proof-of-concept app.

Example:
    python -m android.export \
        --model "aux-ft-all=data/model-aux-ft-all/checkpoints/weights_epoch_11.pt" \
        --model "baseline=data/model-baseline/checkpoints/weights_epoch_20.pt"

Each --model must be NAME=CHECKPOINT_PATH.

For each checkpoint:
    - raw model weights are exported as NAME
    - EMA weights are also exported as NAME (EMA), if present in the checkpoint

Outputs are written directly to:
    android/poc/app/src/main/assets/

The Android app should read:
    misophonia_anc_models.json

and let the user choose between the exported models.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
import typer

from misophonia_anc.model import MisophoniaANCNet

app = typer.Typer(no_args_is_help=True)

CLASS_NAMES = [
    "chewing_gum",
    "clearing_throat",
    "human_breathing",
    "knife_cutting",
    "plastic_crumpling",
    "swallowing",
    "typing",
    "water_drops",
]


class MobileANCStep(nn.Module):
    """
    Fixed-shape mobile inference wrapper.

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
        self.model._subtraction_methods = {}

    def forward(
        self,
        mix: torch.Tensor,
        label: torch.Tensor,
        enc_buf: torch.Tensor,
        dec_buf: torch.Tensor,
        out_buf: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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


def parse_model_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise typer.BadParameter(f"Invalid --model value {spec!r}. Expected format: NAME=CHECKPOINT_PATH.")

    name, checkpoint = spec.split("=", 1)
    name = name.strip()
    checkpoint_path = Path(checkpoint.strip())

    if not name:
        raise typer.BadParameter(f"Invalid --model value {spec!r}: name is empty.")

    if not checkpoint_path.is_file():
        raise typer.BadParameter(f"Checkpoint does not exist: {checkpoint_path}")

    return name, checkpoint_path


def sanitize_asset_stem(name: str) -> str:
    stem = name.lower()
    stem = re.sub(r"\s+", "_", stem)
    stem = re.sub(r"[^a-z0-9_.-]+", "_", stem)
    stem = re.sub(r"_+", "_", stem)
    return stem.strip("_") or "model"


def load_checkpoint(checkpoint_path: Path, *, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "hyperparameters" not in checkpoint:
        raise KeyError(
            f"{checkpoint_path} does not contain 'hyperparameters'. "
            "Expected a checkpoint saved with MisophoniaANCNet.save_checkpoint()."
        )

    if "model_state" not in checkpoint:
        raise KeyError(f"{checkpoint_path} does not contain 'model_state'.")

    return checkpoint


def build_model_from_state(
    checkpoint: dict[str, Any],
    *,
    use_ema: bool,
    device: torch.device,
) -> MisophoniaANCNet:
    model_params = dict(checkpoint["hyperparameters"])
    model = MisophoniaANCNet(**model_params).to(device)

    if use_ema:
        ema_state = checkpoint.get("ema_model_state")
        if ema_state is None:
            raise KeyError("Requested EMA export, but checkpoint does not contain 'ema_model_state'.")
        model.load_state_dict(ema_state)
    else:
        model.load_state_dict(checkpoint["model_state"])

    model.eval()
    model._subtraction_methods = {}
    return model


def validate_same_hyperparameters(
    named_checkpoints: list[tuple[str, Path, dict[str, Any]]],
) -> dict[str, Any]:
    reference_name, _, reference_checkpoint = named_checkpoints[0]
    reference_hparams = dict(reference_checkpoint["hyperparameters"])

    for name, checkpoint_path, checkpoint in named_checkpoints[1:]:
        hparams = dict(checkpoint["hyperparameters"])
        if hparams != reference_hparams:
            raise ValueError(
                "All exported checkpoints must have identical hyperparameters.\n"
                f"Reference: {reference_name}\n"
                f"Mismatch:  {name} ({checkpoint_path})"
            )

    return reference_hparams


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
    *,
    artifacts_dir: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

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
        dynamo=True,
        external_data=False,
        opset_version=18,
        report=True,
        dump_exported_program=True,
        artifacts_dir=str(artifacts_dir),
    )


def verify_with_onnxruntime(
    onnx_path: Path,
    wrapper: MobileANCStep,
    example_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    wrapper.eval()

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


def make_model_metadata(
    *,
    display_name: str,
    checkpoint_path: Path,
    onnx_asset_name: str,
    metadata_asset_name: str,
    model: MisophoniaANCNet,
    example_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    chunk_samples: int,
    sample_rate: int,
    is_ema: bool,
) -> dict[str, Any]:
    mix, label, enc_buf, dec_buf, out_buf = example_inputs

    return {
        "display_name": display_name,
        "onnx_asset_name": onnx_asset_name,
        "metadata_asset_name": metadata_asset_name,
        "checkpoint": str(checkpoint_path),
        "is_ema": is_ema,
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
        "sample_rate": int(sample_rate),
        "label_len": int(model.hyperparameters["label_len"]),
        "class_names": CLASS_NAMES,
        "L": int(model.L),
        "model_dim": int(model.model_dim),
        "lookahead": bool(model.lookahead),
        "note": (
            "Android prototype should keep enc_buf, dec_buf, and out_buf between calls. "
            "The ONNX output is the audio to play back directly. If microphone input is mono, "
            "duplicate it to stereo before feeding 'mix'."
        ),
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


@app.command()
def main(
    models: Annotated[
        list[str],
        typer.Option(
            ...,
            "--model",
            help="Named checkpoint to export. Format: NAME=CHECKPOINT_PATH. Can be repeated.",
        ),
    ],
    *,
    assets_dir: Annotated[
        Path,
        typer.Option(
            ...,
            "--assets-dir",
            help="Android app asset directory where ONNX files and metadata are written.",
        ),
    ] = Path("android/poc/app/src/main/assets"),
    export_dir: Annotated[
        Path,
        typer.Option(
            ...,
            "--export-dir",
            help="Directory for ONNX export reports/artifacts.",
        ),
    ] = Path("android/export"),
    manifest_name: Annotated[
        str,
        typer.Option(..., "--manifest-name", help="Shared Android model manifest asset name."),
    ] = "misophonia_anc_models.json",
    sample_rate: Annotated[
        int,
        typer.Option(..., "--sample-rate", help="Audio sample rate used by the Android app."),
    ] = 44_100,
    label_index: Annotated[
        int,
        typer.Option(
            ...,
            "--label-index",
            help="Example class index used only for ONNX export tracing/verification.",
        ),
    ] = 0,
    chunk_samples: Annotated[
        int | None,
        typer.Option(
            ...,
            "--chunk-samples",
            help="Fixed audio samples per inference step. Default: model.dec_chunk_size * model.L.",
        ),
    ] = None,
    skip_verify: Annotated[
        bool,
        typer.Option(..., "--skip-verify", help="Skip ONNX Runtime numerical verification."),
    ] = False,
) -> None:
    """
    Export one or more checkpoints directly into the Android app assets directory.

    If a checkpoint contains EMA weights, both raw and EMA ONNX models are exported.
    """
    device = torch.device("cpu")
    torch.set_grad_enabled(False)

    assets_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)

    parsed_models = [parse_model_spec(spec) for spec in models]
    named_checkpoints = [
        (name, checkpoint_path, load_checkpoint(checkpoint_path, device=device))
        for name, checkpoint_path in parsed_models
    ]

    validate_same_hyperparameters(named_checkpoints)

    manifest_models: list[dict[str, Any]] = []

    for name, checkpoint_path, checkpoint in named_checkpoints:
        has_ema = checkpoint.get("ema_model_state") is not None

        export_variants = [(name, False)]
        if has_ema:
            export_variants.append((f"{name} (EMA)", True))

        for display_name, use_ema in export_variants:
            stem = sanitize_asset_stem(display_name)
            onnx_asset_name = f"{stem}.onnx"
            metadata_asset_name = f"{stem}.mobile_metadata.json"

            onnx_path = assets_dir / onnx_asset_name
            metadata_path = assets_dir / metadata_asset_name
            artifacts_dir = export_dir / "onnx_export_artifacts" / stem

            model = build_model_from_state(
                checkpoint,
                use_ema=use_ema,
                device=device,
            )

            actual_chunk_samples = (
                int(model.mask_gen.decoder.chunk_size * model.L) if chunk_samples is None else int(chunk_samples)
            )

            wrapper = MobileANCStep(model, chunk_samples=actual_chunk_samples).to(device).eval()

            example_inputs = make_example_inputs(
                model,
                label_index=label_index,
                chunk_samples=actual_chunk_samples,
                device=device,
            )

            print("Export settings:")
            print(f"  display_name:   {display_name}")
            print(f"  checkpoint:     {checkpoint_path}")
            print(f"  output:         {onnx_path}")
            print(f"  use_ema:        {use_ema}")
            print(f"  chunk_samples:  {actual_chunk_samples}")
            print(f"  sample_rate:    {sample_rate}")
            print(f"  label_index:    {label_index}")
            print(f"  label_len:      {model.hyperparameters['label_len']}")
            print(f"  L:              {model.L}")
            print(f"  dec_chunk_size: {model.mask_gen.decoder.chunk_size}")
            print(f"  lookahead:      {model.lookahead}")

            export_onnx(
                wrapper,
                example_inputs,
                onnx_path,
                artifacts_dir=artifacts_dir,
            )
            print(f"Wrote ONNX model: {onnx_path}")

            model_metadata = make_model_metadata(
                display_name=display_name,
                checkpoint_path=checkpoint_path,
                onnx_asset_name=onnx_asset_name,
                metadata_asset_name=metadata_asset_name,
                model=model,
                example_inputs=example_inputs,
                chunk_samples=actual_chunk_samples,
                sample_rate=sample_rate,
                is_ema=use_ema,
            )

            write_json(metadata_path, model_metadata)
            print(f"Wrote metadata: {metadata_path}")

            if not skip_verify:
                verify_with_onnxruntime(onnx_path, wrapper, example_inputs)

            manifest_models.append(model_metadata)

    manifest = {
        "version": 1,
        "sample_rate": int(sample_rate),
        "class_names": CLASS_NAMES,
        "default_class_index": int(label_index),
        "models": manifest_models,
    }

    manifest_path = assets_dir / manifest_name
    write_json(manifest_path, manifest)
    print(f"Wrote Android model manifest: {manifest_path}")


if __name__ == "__main__":
    app()
