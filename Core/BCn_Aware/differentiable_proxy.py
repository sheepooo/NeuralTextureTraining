"""Opaque, linear BC1 blocks. RGB565 endpoints and four-color mode only.

A latent with C channels occupies ceil(C/3) BC1 surfaces; the last surface
replicates its last active channel. Alpha is never used as a latent channel.
"""

import math
import struct
from pathlib import Path

import numpy as np
import torch
from torch import nn


class _ExactSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, source, exact):
        return exact

    @staticmethod
    def backward(ctx, grad):
        return grad, None


def ste_round(x):
    return _ExactSTE.apply(x, x.round())


def pack565(codes):
    codes = codes.long()
    return (codes[..., 0] << 11) | (codes[..., 1] << 5) | codes[..., 2]


def unpack565(words):
    return torch.stack(((words >> 11) & 31, (words >> 5) & 63, words & 31), -1)


def expanded(codes):
    """BC1 bit replication, in integer UNORM8 units."""
    c = codes.long()
    return torch.stack(
        (
            (c[..., 0] << 3) | (c[..., 0] >> 2),
            (c[..., 1] << 2) | (c[..., 1] >> 4),
            (c[..., 2] << 3) | (c[..., 2] >> 2),
        ),
        -1,
    ).float()


class BC1ProxyGrid(nn.Module):
    proxy_kind = 'bc1'

    def __init__(self, endpoints, weights, height, width, channels):
        super().__init__()
        self.endpoints = nn.Parameter(endpoints.float())
        # Linear palette positions 0,1,2,3 map to BC1 selectors 0,2,3,1.
        self.weights = nn.Parameter(weights.float())
        self.register_buffer(
            "shape", torch.tensor([height, width, channels], device=endpoints.device)
        )
        self._image_shape = (height, width)
        self.channels = channels

    @classmethod
    @torch.no_grad()
    def from_image(cls, image):
        if image.ndim != 3 or not 1 <= image.shape[-1] <= 4:
            raise ValueError("BC1 latent image must have 1 to 4 channels")
        h, w, channels = image.shape
        bh, bw = math.ceil(h / 4), math.ceil(w / 4)
        y = torch.arange(bh * 4, device=image.device).clamp(max=h - 1)
        x = torch.arange(bw * 4, device=image.device).clamp(max=w - 1)
        padded = image[y[:, None], x[None, :]].clamp(0, 1)
        blocks = []
        for start in range(0, channels, 3):
            rgb = padded[..., start : start + 3]
            rgb = torch.cat((rgb, rgb[..., -1:].expand(*rgb.shape[:-1], 3 - rgb.shape[-1])), -1)
            blocks.append(rgb.reshape(bh, 4, bw, 4, 3).permute(0, 2, 1, 3, 4).reshape(-1, 16, 3))
        pixels = torch.cat(blocks)
        # PCA seed avoids choosing unrelated per-component min/max endpoints.
        centered = pixels - pixels.mean(1, keepdim=True)
        covariance = centered.transpose(1, 2) @ centered
        direction = torch.ones_like(pixels[:, 0])
        for _ in range(8):
            direction = (covariance @ direction[..., None]).squeeze(-1)
            direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        scores = (centered * direction[:, None]).sum(-1)
        row = torch.arange(len(pixels), device=image.device)
        ends = torch.stack((pixels[row, scores.argmax(1)], pixels[row, scores.argmin(1)]), 1)
        scale = image.new_tensor([31, 63, 31])
        codes = (ends * scale).round()
        packed = pack565(codes)
        hi, lo = packed.max(1).values, packed.min(1).values
        hi = torch.where(hi == lo, (hi + 1).clamp(max=65535), hi)
        lo = torch.where(hi == lo, (lo - 1).clamp(min=0), lo)
        codes = unpack565(torch.stack((hi, lo), 1)).float()
        a, b = expanded(codes).unbind(1)
        palette = (
            torch.stack((a, torch.floor((2 * a + b) / 3), torch.floor((a + 2 * b) / 3), b), 1) / 255
        )
        weights = ((pixels[:, :, None] - palette[:, None]) ** 2).sum(-1).argmin(-1)
        return cls(codes, weights, h, w, channels)

    def _legal_endpoints(self):
        scale = self.endpoints.new_tensor([31, 63, 31])
        codes = ste_round(self.endpoints.clamp(min=0).minimum(scale))
        words = pack565(codes.detach())
        hi, lo = words[:, 0], words[:, 1]
        # Keep endpoint identities stable for optimizer momentum. Project crossings
        # into opaque four-color mode; never silently switch to transparent BC1.
        hi = torch.maximum(hi, lo + 1).clamp(max=65535)
        lo = torch.minimum(lo, hi - 1).clamp(min=0)
        legal = unpack565(torch.stack((hi, lo), 1)).to(codes)
        return _ExactSTE.apply(codes, legal)

    def decode_image(self):
        codes = self._legal_endpoints()
        linear = codes * codes.new_tensor([255 / 31, 255 / 63, 255 / 31])
        rgb = _ExactSTE.apply(linear, expanded(codes.detach()))
        a, b = rgb.unbind(1)
        k = ste_round(self.weights.clamp(0, 3)) / 3
        values = a[:, None] * (1 - k[..., None]) + b[:, None] * k[..., None]
        # Integer division is part of the BC1 decode contract.
        values = _ExactSTE.apply(values, torch.floor(values.detach() + 1e-5))
        h, w = self._image_shape
        bh, bw = math.ceil(h / 4), math.ceil(w / 4)
        surfaces = (
            values.reshape(-1, bh, bw, 4, 4, 3)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(-1, bh * 4, bw * 4, 3)
        )
        image = torch.cat(list(surfaces[:, :h, :w].unbind(0)), -1)[..., : self.channels]
        return image / 255

    @torch.no_grad()
    def clamp_parameters(self):
        # Preserve sub-code residuals so Adam can cross quantization thresholds.
        # Rounding here would erase every small endpoint update after each step.
        limits = self.endpoints.new_tensor([31, 63, 31])
        self.endpoints.clamp_(min=0)
        self.endpoints.copy_(torch.minimum(self.endpoints, limits))
        self.weights.clamp_(0, 3)

    @torch.no_grad()
    def export_blocks(self):
        ends = pack565(self._legal_endpoints()).cpu().numpy().astype("<u2")
        selectors = torch.tensor([0, 2, 3, 1], device=self.weights.device)[
            self.weights.round().long().clamp(0, 3)
        ]
        bits = (
            (selectors.long() << (2 * torch.arange(16, device=selectors.device)))
            .sum(-1)
            .cpu()
            .numpy()
            .astype("<u4")
        )
        result = bytearray()
        for endpoints, indices in zip(ends, bits):
            result.extend(struct.pack("<HHI", int(endpoints[0]), int(endpoints[1]), int(indices)))
        h, w = self._image_shape
        surface_bytes = math.ceil(h / 4) * math.ceil(w / 4) * 8
        return [bytes(result[i : i + surface_bytes]) for i in range(0, len(result), surface_bytes)]

    def export_dds(self, directory, grid_index):
        h, w = self._image_shape
        for index, payload in enumerate(self.export_blocks()):
            Path(directory, f"feature_grid_{grid_index}_bc1_{index}.dds").write_bytes(
                dds_header(w, h) + payload
            )

    @torch.no_grad()
    def decode_cpu_image(self):
        # Independent stream decoder, also usable without torch/CUDA extensions.
        h, w = self._image_shape
        images = [decode_blocks(data, w, h) for data in self.export_blocks()]
        return (
            torch.from_numpy(np.concatenate(images, -1)[..., : self.channels].copy()).float() / 255
        )

    @classmethod
    def from_state(cls, state, prefix, device):
        h, w, c = state[prefix + "shape"].tolist()
        return cls(
            state[prefix + "endpoints"].to(device), state[prefix + "weights"].to(device), h, w, c
        )


from Common.Block_Compression.bc import dds_header, decode_blocks
