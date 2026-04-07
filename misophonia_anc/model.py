"""
Main model definiton.

Heavily based on https://github.com/vb000/SemanticHearing
"""
# ruff: noqa: ANN001, ANN002, ANN003 # TODO: Improve quality

from pathlib import Path

import eliot
import torch
import torch.nn as nn

from ._utils import MisophoniaANCConfig, get_git_sha, mod_pad
from .decoder import CausalTransformerDecoder
from .encoder import DilatedCausalConvEncoder


class MisophoniaANCNet(nn.Module):
    def __init__(
        self,
        label_len,
        *,
        L=8,  # noqa: N803 # TODO: Improve name?
        model_dim=128,  # Original 512
        num_enc_layers=10,
        dec_buf_len=100,
        num_dec_layers=2,
        dec_chunk_size=72,
        out_buf_len=2,
        use_pos_enc=True,
        conditioning="mult",
        lookahead=True,
        pretrained_path=None,
    ) -> None:
        super(MisophoniaANCNet, self).__init__()

        self._hyperparameters = {
            "label_len": label_len,
            "L": L,
            "model_dim": model_dim,
            "num_enc_layers": num_enc_layers,
            "dec_buf_len": dec_buf_len,
            "num_dec_layers": num_dec_layers,
            "dec_chunk_size": dec_chunk_size,
            "out_buf_len": out_buf_len,
            "use_pos_enc": use_pos_enc,
            "conditioning": conditioning,
            "lookahead": lookahead,
        }

        self.L = L
        self.out_buf_len = out_buf_len
        self.model_dim = model_dim
        self.lookahead = lookahead

        # Input conv to convert input audio to a latent representation
        kernel_size = 3 * L if lookahead else L
        self.in_conv = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=model_dim, kernel_size=kernel_size, stride=L, padding=0, bias=False),
            nn.ReLU(),
        )

        # Label embedding layer
        self.label_embedding = nn.Sequential(
            nn.Linear(label_len, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, model_dim),
            nn.LayerNorm(model_dim),
            nn.ReLU(),
        )

        # Mask generator
        self.mask_gen = MaskNet(
            model_dim=model_dim,
            num_enc_layers=num_enc_layers,
            dec_buf_len=dec_buf_len,
            dec_chunk_size=dec_chunk_size,
            num_dec_layers=num_dec_layers,
            use_pos_enc=use_pos_enc,
            conditioning=conditioning,
        )

        # Output conv layer
        self.out_conv = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=model_dim,
                out_channels=2,
                kernel_size=(out_buf_len + 1) * L,
                stride=L,
                padding=out_buf_len * L,
                bias=False,
            ),
            nn.Tanh(),
        )

        if pretrained_path is not None:
            # TODO: I think this should not be done in the constructor and does not need to be concerned about loading the checkpoint
            state_dict = torch.load(pretrained_path)["model_state"]

            # Load all the layers except label_embedding and freeze them
            for name, param in self.named_parameters():
                if "label_embedding" not in name:
                    param.data = state_dict[name]
                    param.requires_grad = False

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

        if init_enc_buf is None:
            return out
        else:
            return out, enc_buf, dec_buf, out_buf

    #### UTILITY FUNCTIONS ####
    @property
    def hyperparameters(self) -> dict:
        """Get the hyperparameters used to initialize the model. This can be useful for logging and checkpointing."""
        return dict(self._hyperparameters)

    def save_checkpoint(self, ckpt_path: Path, **other_info: dict) -> None:
        """
        Save model checkpoint.

        Args:
            ckpt_path: Path to save the checkpoint file.
            epoch: Current epoch number.
            val_si_snr_improvement: Validation SI-SNR improvement at the current epoch.
            hyperparameters: Dictionary of model hyperparameters to save in the checkpoint.
        """
        torch.save(
            {
                "model_state": self.state_dict(),
                "hyperparameters": self.hyperparameters,
                "git_sha": get_git_sha(),
                **other_info,
            },
            ckpt_path,
        )

    @classmethod
    def from_config(
        cls, config: MisophoniaANCConfig, *, checkpoint: Path | None = None, device: torch.device | None = None
    ) -> "MisophoniaANCNet":
        """
        Load model from config and checkpoint.

        Args:
            config: MisophoniaANCConfig containing model hyperparameters.
            checkpoint: Optional path to a checkpoint to load model weights from.
                            If None, model will be initialized with random weights.
            device: Device to move the model to. If None, the model will be moved to the default device.

        Returns:
            An instance of MisophoniaANCNet initialized according to the provided config and checkpoint.
        """
        model_params = dict(config.model_params)

        if checkpoint is not None:
            checkpoint = Path(checkpoint)
            assert checkpoint.is_file(), f"Checkpoint path {checkpoint} does not exist or is not a file."
            checkpoint_data = torch.load(checkpoint, map_location=device)
            assert "model_state" in checkpoint_data, f"Checkpoint file {checkpoint} does not contain 'model_state'."

            if "hyperparameters" in checkpoint_data:
                for key, value in model_params.items():
                    if key in checkpoint_data["hyperparameters"] and checkpoint_data["hyperparameters"][key] != value:
                        eliot.log_message(
                            f"Checkpoint hyperparameter {key} has value {value} which does not match config value {model_params[key]}. Replacing config value with checkpoint value.",
                            level="warning",
                        )
                        model_params[key] = value
            else:
                eliot.log_message(f"Checkpoint {checkpoint} does not contain hyperparameters.", level="warning")

            model = MisophoniaANCNet(**model_params)
            state_dict = checkpoint_data["model_state"]
            model.load_state_dict(state_dict)

        if device is not None:
            model.to(device)

        return model


class MaskNet(nn.Module):
    def __init__(
        self, model_dim, num_enc_layers, dec_buf_len, dec_chunk_size, num_dec_layers, use_pos_enc, conditioning
    ) -> None:
        super(MaskNet, self).__init__()

        # Encoder based on dilated causal convolutions.
        self.encoder = DilatedCausalConvEncoder(channels=model_dim, num_layers=num_enc_layers)

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
        m, dec_buf = self.decoder(input=e, embedding=l, ctx_buf=dec_buf)

        return m, enc_buf, dec_buf
