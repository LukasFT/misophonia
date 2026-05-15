"""
Main model definiton.

Heavily based on https://github.com/vb000/SemanticHearing
"""
# ruff: noqa: ANN001, ANN002, ANN003 # TODO: Improve quality

import copy
from pathlib import Path

import eliot
import mlflow
import numpy as np
import torch
import torch.nn as nn

from ._utils import GtTargets, MisophoniaANCConfig, get_git_sha, mod_pad
from .decoder import CausalTransformerDecoder
from .encoder import DilatedCausalConvEncoder
from .subtraction_methods import SubtractionMethod, ls_fir_subtraction, simple_subtraction, stft_subtraction


class MisophoniaANCNet(nn.Module):
    def __init__(
        self,
        label_len,
        *,
        L=8,  # noqa: N803 # TODO: Improve name?
        model_dim=128,  # Original 512
        audio_channels=2,
        num_enc_layers=10,
        dec_buf_len=100,
        num_dec_layers=2,
        dec_chunk_size=72,
        out_buf_len=2,
        use_pos_enc=True,
        conditioning="mult",
        lookahead=True,
        dropout_label: float | None = None,  # None for backwards compatibility (equivalent to 0.0)
        dropout_encoder: float | None = None,  # None for backwards compatibility (equivalent to 0.0)
        dropout_decoder: float = 0.1,  # 0.1 is used by torch.nn.TransformerDecoderLayer, which Semantic Hearing directly used
        dropout_pos: float = 0.0,  # There was always a droput layer, but it was deafault set to 0.0
        ground_truth_target: GtTargets = "isolated_trigger",
        decoder_batches_parallel_k: int = 4000,
        gt_is_isolated_trigger=None,  # For backwards compatibility, use ground_truth_target instead
    ) -> None:
        super(MisophoniaANCNet, self).__init__()

        assert ground_truth_target in ["isolated_trigger", "clean_mix"]

        if gt_is_isolated_trigger is not None:
            eliot.log_message(
                f"gt_is_isolated_trigger is deprecated and will be removed in a future version. Using it to override ground_truth_target (given {gt_is_isolated_trigger=}, overrides {ground_truth_target=}).",
                level="warning",
            )
            if gt_is_isolated_trigger:
                ground_truth_target = "isolated_trigger"
            else:
                ground_truth_target = "clean_mix"

        self._hyperparameters = {
            "label_len": label_len,
            "L": L,
            "model_dim": model_dim,
            "audio_channels": audio_channels,
            "num_enc_layers": num_enc_layers,
            "dec_buf_len": dec_buf_len,
            "num_dec_layers": num_dec_layers,
            "dec_chunk_size": dec_chunk_size,
            "out_buf_len": out_buf_len,
            "use_pos_enc": use_pos_enc,
            "conditioning": conditioning,
            "lookahead": lookahead,
            "dropout_label": dropout_label,
            "dropout_encoder": dropout_encoder,
            "dropout_decoder": dropout_decoder,
            "dropout_pos": dropout_pos,
            "ground_truth_target": ground_truth_target,
            #  # Only affects speed, not the result, so do not store decoder_batches_parallel_k as hyperparameter
            # "decoder_batches_parallel_k": decoder_batches_parallel_k,
        }

        self.L = L
        self.out_buf_len = out_buf_len
        self.model_dim = model_dim
        self.lookahead = lookahead

        # Input conv to convert input audio to a latent representation
        kernel_size = 3 * L if lookahead else L
        self.in_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=audio_channels,
                out_channels=model_dim,
                kernel_size=kernel_size,
                stride=L,
                padding=0,
                bias=False,
            ),
            nn.ReLU(),
        )

        # Label embedding layer
        label_embedding_layers = [
            nn.Linear(label_len, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
        ]
        if dropout_label is not None:
            label_embedding_layers.append(nn.Dropout(p=dropout_label))
        label_embedding_layers.extend(
            [
                nn.Linear(512, model_dim),
                nn.LayerNorm(model_dim),
                nn.ReLU(),
            ]
        )
        self.label_embedding = nn.Sequential(*label_embedding_layers)

        # Mask generator
        self.mask_gen = MaskNet(
            model_dim=model_dim,
            num_enc_layers=num_enc_layers,
            dec_buf_len=dec_buf_len,
            dec_chunk_size=dec_chunk_size,
            num_dec_layers=num_dec_layers,
            use_pos_enc=use_pos_enc,
            conditioning=conditioning,
            decoder_batches_parallel_k=decoder_batches_parallel_k,
            dropout_encoder=dropout_encoder,
            dropout_decoder=dropout_decoder,
            dropout_pos=dropout_pos,
        )

        # Output conv layer
        self.out_conv = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=model_dim,
                out_channels=audio_channels,
                kernel_size=(out_buf_len + 1) * L,
                stride=L,
                padding=out_buf_len * L,
                bias=False,
            ),
            nn.Tanh(),
        )

        self._subtraction_methods: dict[str, SubtractionMethod] = {}

    def init_buffers(self, batch_size, device):  # noqa: ANN201
        enc_buf = self.mask_gen.encoder.init_ctx_buf(batch_size, device)
        dec_buf = self.mask_gen.decoder.init_ctx_buf(batch_size, device)
        out_buf = torch.zeros(batch_size, self.model_dim, self.out_buf_len, device=device)
        return enc_buf, dec_buf, out_buf

    def predict(self, x, label, enc_buf, dec_buf, out_buf):  # noqa: ANN201
        """Generate latent space representation of the input"""
        x = self.in_conv(x)

        # Generate label embedding
        l = self.label_embedding(label)  # [B, label_len] --> [B, channels]
        l = l.unsqueeze(1).unsqueeze(-1)  # [B, 1, channels, 1]

        # Generate mask corresponding to the label
        m, enc_buf, dec_buf = self.mask_gen(x, l, enc_buf, dec_buf)

        # Apply mask and decode
        x = x * m
        x = torch.cat((out_buf, x), dim=-1)
        out_buf = x[..., -self.out_buf_len :]
        x = self.out_conv(x)

        return x, enc_buf, dec_buf, out_buf

    def forward(  # noqa: ANN201
        self,
        inputs,
        *,
        init_enc_buf=None,
        init_dec_buf=None,
        init_out_buf=None,
        pad=True,
        # TODO: The below are unused?
        writer=None,
        step=None,
        idx=None,
    ):
        """
        Extracts the audio corresponding to the `label` in the given
        `mixture`. Generates `chunk_size` samples per iteration.
        Args:
            mixed: [B, n_mics, T]
                input audio mixture
            label: [B, num_labels]
                one hot label
        Returns:
            out: [B, n_spk, T]
                extracted audio with sounds corresponding to the `label`
        """
        x, label = inputs["mix"], inputs["label_vector"]

        if init_enc_buf is None or init_dec_buf is None or init_out_buf is None:
            assert init_enc_buf is None and init_dec_buf is None and init_out_buf is None, (
                "Both buffers have to initialized, or both of them have to be None."
            )
            enc_buf, dec_buf, out_buf = self.init_buffers(x.shape[0], x.device)
        else:
            enc_buf, dec_buf, out_buf = init_enc_buf, init_dec_buf, init_out_buf

        mod = 0
        if pad:
            pad_size = (self.L, self.L) if self.lookahead else (0, 0)
            x, mod = mod_pad(x, chunk_size=self.L, pad=pad_size)

        x, enc_buf, dec_buf, out_buf = self.predict(x, label, enc_buf, dec_buf, out_buf)

        # Remove mod padding, if present.
        if mod != 0:
            x = x[:, :, :-mod]

        out = {"x": x}

        for subtraction_name, subtraction_func in self._subtraction_methods.items():
            out[f"x_{subtraction_name}"] = subtraction_func(inputs["mix"], x)

        if init_enc_buf is None:
            return out
        else:
            return out, enc_buf, dec_buf, out_buf

    def register_subtraction_method(self, name: str, func: SubtractionMethod | None = None) -> None:
        """
        Register a subtraction method to be applied to the model prediction.

        Args:
            name: The name of the subtraction method (e.g. "simple", "stft", "ls_fir").
            func: A function that takes in the mixture audio and the model prediction and returns the subtracted audio.
                    If None, the method will be looked up in the model's internal dictionary of known subtraction methods.

        """
        assert self._hyperparameters["ground_truth_target"] == "isolated_trigger", (
            "Subtraction can only be applied if ground_truth_target is 'isolated_trigger'"
        )
        known_methods = {
            "simple": simple_subtraction,
            "stft": stft_subtraction,
            "ls_fir": ls_fir_subtraction,
        }
        func = func if func is not None else known_methods.get(name)
        if func is None:
            raise ValueError(f"Subtraction method {name} is not recognized and no function was provided.")
        self._subtraction_methods[name] = func

    #### UTILITY FUNCTIONS ####
    @property
    def hyperparameters(self) -> dict:
        """Get the hyperparameters used to initialize the model. This can be useful for logging and checkpointing."""
        return dict(self._hyperparameters)

    @property
    def ground_truth_target(self) -> GtTargets:
        """Get the ground truth target type used for training."""
        return self._hyperparameters["ground_truth_target"]

    def save_checkpoint(
        self,
        ckpt_path: Path,
        *,
        epoch: int,
        global_step_train: int,
        global_step_val: int,
        ema_model: "ModelEMA | None" = None,
        **other_info: dict,
    ) -> None:
        """
        Save model checkpoint.

        Args:
            ckpt_path: Path to save the checkpoint file.
            epoch: Current epoch number.
            global_step_train: Total number of training batches logged.
            global_step_val: Total number of validation batches logged.
            ema_model: Optional EMA model to include in the checkpoint.
            other_info: Additional key-value pairs to include in the checkpoint metadata (e.g. metrics like val_loss, val_si_snr_improvement).
        """
        torch.save(
            {
                "model_state": self.state_dict(),
                "hyperparameters": self.hyperparameters,
                "epoch": epoch,
                "git_sha": get_git_sha(),
                "mlflow_run_id": mlflow.active_run().info.run_id if mlflow.active_run() is not None else None,
                "global_step_train": global_step_train,
                "global_step_val": global_step_val,
                "ema_model_state": ema_model.model.state_dict() if ema_model is not None else None,
                "ema_decay": ema_model.decay if ema_model is not None else None,
                **other_info,
            },
            ckpt_path,
        )

    @classmethod
    def from_config(
        cls,
        config: MisophoniaANCConfig,
        *,
        checkpoint: Path | None = None,
        device: torch.device | None = None,
    ) -> tuple["MisophoniaANCNet", dict]:
        """
        Load model from config and checkpoint.

        Args:
            config: MisophoniaANCConfig containing model hyperparameters.
            checkpoint: Optional path to a checkpoint to load model weights from.
                            If None, model will be initialized with random weights.
            device: Device to move the model to. If None, the model will be moved to the default device.

        Returns:
            a tuple containing:
             - An instance of MisophoniaANCNet initialized according to the provided config and checkpoint.
             - A dictionary containing metadata from the checkpoint (e.g. epoch, hyperparameters) if a checkpoint was provided, or an empty dictionary if no checkpoint was provided.
        """
        model_params = dict(config.model_params)
        metadata = {}

        if checkpoint is None:
            model = MisophoniaANCNet(**model_params)
            metadata["epoch"] = 0

            if config.ema_decay is not None:
                metadata["ema_model"] = ModelEMA(model, decay=config.ema_decay)

        else:
            checkpoint = Path(checkpoint)
            assert checkpoint.is_file(), f"Checkpoint path {checkpoint} does not exist or is not a file."
            metadata = torch.load(checkpoint, map_location=device)
            assert "model_state" in metadata, f"Checkpoint file {checkpoint} does not contain 'model_state'."
            assert "epoch" in metadata, f"Checkpoint file {checkpoint} does not contain 'epoch'."

            if "hyperparameters" in metadata:
                for key, value in model_params.items():
                    if key in metadata["hyperparameters"] and metadata["hyperparameters"][key] != value:
                        eliot.log_message(
                            f"Checkpoint hyperparameter {key} has value {value} which does not match config value {model_params[key]}. Replacing config value with checkpoint value.",
                            level="warning",
                        )
                        model_params[key] = value
                for key in metadata["hyperparameters"]:
                    if key not in model_params:
                        # Set hyperparameters which are present in the checkpoint but not in the config.
                        # This allows for loading checkpoints which were trained with an older version of the code which had different default hyperparameters.
                        model_params[key] = metadata["hyperparameters"][key]

            else:
                eliot.log_message(f"Checkpoint {checkpoint} does not contain hyperparameters.", level="warning")

            model = MisophoniaANCNet(**model_params)
            state_dict = metadata["model_state"]
            metadata.pop("model_state")  # Remove model state from metadata to avoid confusion
            model.load_state_dict(state_dict)

            # Load EMA:
            ema_state = metadata.pop("ema_model_state")  # Remove EMA state from metadata to avoid confusion
            ema_decay = metadata.pop("ema_decay")  # Remove EMA decay from metadata to avoid confusion
            assert (ema_state is None) == (ema_decay is None), (
                "EMA state and decay must both be present or both be None in the checkpoint."
            )
            if ema_state is not None and ema_decay is not None:
                if not np.isclose(ema_decay, config.ema_decay):
                    eliot.log_message(
                        f"EMA decay in checkpoint ({ema_decay}) does not match EMA decay in config ({config.ema_decay}). Using EMA decay from config.",
                        level="warning",
                    )
                    ema_decay = config.ema_decay
                metadata["ema_model"] = ModelEMA(model, decay=ema_decay, model_state=ema_state)
                metadata["ema_model"].to(device) if device is not None else None

        if config.subtraction_methods is not None:
            for method_name in config.subtraction_methods:
                model.register_subtraction_method(
                    name=method_name,
                    func=None,  # Automatically look up the function from the name
                )

        if device is not None:
            model.to(device)

        return model, metadata


class MaskNet(nn.Module):
    def __init__(
        self,
        *,
        model_dim,
        num_enc_layers,
        dec_buf_len,
        dec_chunk_size,
        num_dec_layers,
        use_pos_enc,
        conditioning,
        dropout_encoder: float,
        dropout_decoder: float,
        dropout_pos: float,
        decoder_batches_parallel_k: int = 4000,
    ) -> None:
        super(MaskNet, self).__init__()

        self._decoder_batches_parallel_k = decoder_batches_parallel_k

        # Encoder based on dilated causal convolutions.
        self.encoder = DilatedCausalConvEncoder(
            channels=model_dim,
            num_layers=num_enc_layers,
            dropout=dropout_encoder,
        )

        # Transformer decoder that operates on chunks of size
        # buffer size.
        self.decoder = CausalTransformerDecoder(
            model_dim=model_dim,
            ctx_len=dec_buf_len,
            chunk_size=dec_chunk_size,
            num_layers=num_dec_layers,
            nhead=8,
            use_pos_enc=use_pos_enc,
            ff_dim=2 * model_dim,
            conditioning=conditioning,
            dropout_decoder=dropout_decoder,
            dropout_pos=dropout_pos,
        )

    def forward(self, x, l, enc_buf, dec_buf):  # noqa: ANN201
        """
        Generates a mask based on encoded input `e` and the one-hot
        label `label`.

        Args:
            x: [B, C, T]
                Input audio sequence
            l: [B, C]
                Label embedding
            ctx_buf: {[B, C, <receptive field of the layer>], ...}
                List of context buffers maintained by DCC encoder
        """
        # Enocder the label integrated input
        e, enc_buf = self.encoder(x, enc_buf)

        # Decoder conditioned on embedding
        m, dec_buf = self.decoder(
            input=e,
            embedding=l,
            ctx_buf=dec_buf,
            K=self._decoder_batches_parallel_k,
        )

        return m, enc_buf, dec_buf


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999, model_state: dict | None = None) -> None:
        self.model = copy.deepcopy(model).eval()
        if model_state is not None:
            self.model.load_state_dict(model_state)
        self.decay = decay

        for p in self.model.parameters():
            p.requires_grad_(False)  # noqa: FBT003

    def __call__(self, *args, **kwargs):  # noqa: ANN204
        return self.model(*args, **kwargs)

    def __getattr__(self, name):  # noqa: ANN204
        return getattr(self.model, name)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        ema_state = self.model.state_dict()
        model_state = model.state_dict()

        for key, model_value in model_state.items():
            ema_value = ema_state[key]

            if torch.is_floating_point(model_value):
                ema_value.mul_(self.decay).add_(model_value, alpha=1.0 - self.decay)
            else:
                ema_value.copy_(model_value)

    def to(self, device: torch.device) -> "ModelEMA":
        self.model.to(device)
        return self
