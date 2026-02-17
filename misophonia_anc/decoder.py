"""
Decoder logic.

Heavily based on https://github.com/vb000/SemanticHearing
"""

# ruff: noqa: ANN001 # TODO: Improve quality

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from speechbrain.lobes.models.transformer.Transformer import PositionalEncoding
from torch import Tensor

from ._utils import mod_pad


class CausalTransformerDecoder(nn.Module):
    """
    A casual transformer decoder which decodes input vectors using
    precisely `ctx_len` past vectors in the sequence, and using no future
    vectors at all.
    """

    def __init__(
        self, model_dim, ctx_len, chunk_size, num_layers, nhead, use_pos_enc, ff_dim, conditioning="conv"
    ) -> None:
        super(CausalTransformerDecoder, self).__init__()
        self.num_layers = num_layers
        self.model_dim = model_dim
        self.ctx_len = ctx_len
        self.chunk_size = chunk_size
        self.nhead = nhead
        self.use_pos_enc = use_pos_enc
        self.unfold = nn.Unfold(kernel_size=(ctx_len + chunk_size, 1), stride=chunk_size)
        self.pos_enc_tgt = PositionalEncoding(model_dim, max_len=1000)
        self.pos_enc_mem = PositionalEncoding(model_dim, max_len=100)
        self.tf_dec_layers = nn.ModuleList(
            [
                _CausalTransformerDecoderLayer(d_model=model_dim, nhead=nhead, dim_feedforward=ff_dim, batch_first=True)
                for _ in range(num_layers)
            ]
        )
        self.conditioning = conditioning

        if conditioning == "film":
            self.film = nn.Sequential(nn.Linear(model_dim, 2 * model_dim), nn.ReLU())

    def init_ctx_buf(self, batch_size, device) -> Tensor:
        return torch.zeros((batch_size, self.num_layers + 1, self.ctx_len, self.model_dim), device=device)

    def _causal_unfold(self, x) -> Tensor:
        """
        Unfolds the sequence into a batch of sequences
        prepended with `ctx_len` previous values.

        Args:
            x: [B, ctx_len + L, C]
            ctx_len: int
        Returns:
            [B * L, ctx_len + 1, C]
        """
        B, T, C = x.shape  # noqa: N806
        x = x.permute(0, 2, 1)  # [B, C, ctx_len + L]
        x = self.unfold(x.unsqueeze(-1))  # [B, C * (ctx_len + chunk_size), -1]
        x = x.permute(0, 2, 1)
        x = x.reshape(B, -1, C, self.ctx_len + self.chunk_size)
        x = x.reshape(-1, C, self.ctx_len + self.chunk_size)
        x = x.permute(0, 2, 1)
        return x

    def forward(self, input, embedding, ctx_buf, K=4000):  # noqa: ANN201, N803
        """
        Args:
            input: [B, model_dim, T]
            embedding: [B, NE, model_dim, embed_len]
            ctx_buf: [B, num_layers, ctx_len, model_dim]
            K: int
                Number of batches to process at once to avoid OOM.
        Returns:
            output: [B, model_dim, T]
            ctx_buf: [B, num_layers, ctx_len, model_dim]
        """

        # Mod pad the input so the sequence length is a multiple
        # of chunk_size.
        input, mod = mod_pad(input, self.chunk_size, (0, 0))

        # Init
        B, C, T = input.shape  # noqa: N806
        output = input.permute(0, 2, 1).contiguous()
        mem = None

        if self.conditioning == "conv":
            # Convolutional/mutltiplicative conditioning
            input = input.view(1, B * C, T)
            input = F.pad(input, (embedding.shape[-1] - 1, 0))  # [1, B * C, T + embed_len - 1]
            emb_filter = torch.mean(embedding, dim=1).reshape(B * C, 1, -1)
            output = F.conv1d(input, emb_filter, groups=B * C)
            output = output.view(B, C, T)
            output = output.permute(0, 2, 1)
        elif self.conditioning == "attn":
            # Use cross attn for conditioning
            mem = embedding.permute(0, 1, 3, 2)  # [B, NE, embed_len, C]
            if self.use_pos_enc:
                mem = mem.view(-1, mem.shape[-2], mem.shape[-1])
                mem = mem + self.pos_enc_mem(mem)
                mem = mem.view(B, -1, mem.shape[-2], mem.shape[-1])
            mem = mem.reshape(B, -1, mem.shape[-1])  # [B, NE * embed_len, C]
            mem = mem.unsqueeze(1).repeat(1, (T // self.chunk_size), 1, 1)  # [B, T // chunk_size, NE * embed_len, C]
            mem = mem.reshape(-1, mem.shape[-2], mem.shape[-1])  # [B * (T // chunk_size), NE * embed_len, C]
        elif self.conditioning == "film":
            # Use FILM for conditioning
            emb_filter = torch.mean(embedding, dim=(1, 3))  # [B, C]
            emb_filter = self.film(emb_filter)  # [B, 2 * C]
            gamma, beta = emb_filter.chunk(2, dim=-1)
            output = output * gamma.unsqueeze(1) + beta.unsqueeze(1)
        else:
            emb_filter = torch.mean(embedding, dim=(1, 3))  # [B, C]
            output = output * emb_filter.unsqueeze(1)  # [B, T, C]

        for i, layer in enumerate(self.tf_dec_layers):
            # Prepend the context to the input and update the context
            # [B, ctx_len + T, C]
            tgt = torch.cat([ctx_buf[:, i, :, :], output], dim=1)
            ctx_buf[:, i, :, :] = tgt[:, -self.ctx_len :, :]

            # Unfold the sequence into a batch of sequences prepended
            # with `ctx_len` previous values.
            # [B * (T // chunk_size), ctx_len + chunk_size, C]
            tgt = self._causal_unfold(tgt)

            # Positional encoding
            if i == 0 and self.use_pos_enc:
                tgt = tgt + self.pos_enc_tgt(tgt)

            _tgt = torch.zeros_like(tgt)[:, : self.chunk_size, :]
            for k in range(int(math.ceil(tgt.shape[0] / K))):
                s, e = k * K, (k + 1) * K
                _mem = None if mem is None else mem[s:e]
                _tgt[s:e], _, _ = layer(tgt[s:e], _mem, self.chunk_size)

            output = _tgt.reshape(B, T, C)

        # Remove the mod padding
        output = output.permute(0, 2, 1)
        if mod != 0:
            output = output[:, :, :-mod]

        return output, ctx_buf


class _CausalTransformerDecoderLayer(torch.nn.TransformerDecoderLayer):
    """
    Adapted from:
    "https://github.com/alexmt-scale/causal-transformer-decoder/blob/"
    "0caf6ad71c46488f76d89845b0123d2550ef792f/"
    "causal_transformer_decoder/model.py#L77"
    """

    def forward(self, tgt: Tensor, memory: Optional[Tensor] = None, chunk_size: int = 1) -> Tensor:
        tgt_last_tok = tgt[:, -chunk_size:, :]

        # self attention part
        tmp_tgt, sa_map = self.self_attn(
            tgt_last_tok,
            tgt,
            tgt,
            attn_mask=None,  # not needed because we only care about the last token
            key_padding_mask=None,
        )
        tgt_last_tok = tgt_last_tok + self.dropout1(tmp_tgt)
        tgt_last_tok = self.norm1(tgt_last_tok)

        # encoder-decoder attention
        ca_map = None
        if memory is not None:
            tmp_tgt, ca_map = self.multihead_attn(
                tgt_last_tok,
                memory,
                memory,
                attn_mask=None,  # Attend to the entire chunk
                key_padding_mask=None,
            )
            tgt_last_tok = tgt_last_tok + self.dropout2(tmp_tgt)
            tgt_last_tok = self.norm2(tgt_last_tok)

        # final feed-forward network
        tmp_tgt = self.linear2(self.dropout(self.activation(self.linear1(tgt_last_tok))))
        tgt_last_tok = tgt_last_tok + self.dropout3(tmp_tgt)
        tgt_last_tok = self.norm3(tgt_last_tok)
        return tgt_last_tok, sa_map, ca_map
