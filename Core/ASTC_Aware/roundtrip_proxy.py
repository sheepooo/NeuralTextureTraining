"""ASTC codec-aware identity STE with explicit, stable encoded state."""

import numpy as np
import torch
from torch import nn
from Common.Block_Compression.astc import ASTCCodec


class ASTCRoundtripGrid(nn.Module):
    proxy_kind = "astc_roundtrip"

    def __init__(self, image, codec):
        super().__init__()
        self.endpoints = nn.Parameter(image.float().clone())
        self.register_buffer("shape", torch.tensor(image.shape, device=image.device))
        self.register_buffer(
            "footprint",
            torch.tensor(list(map(int, codec.astc_block.split("x"))), device=image.device),
        )
        self.register_buffer("bitstream", torch.empty(0, dtype=torch.uint8, device=image.device))
        self.register_buffer(
            "encoded_source", torch.empty(0, dtype=torch.uint8, device=image.device)
        )
        self.register_buffer("codec_frozen", torch.tensor(False, device=image.device))
        self._image_shape = tuple(image.shape[:2])
        self.codec = codec
        self._cache = None

    def _source(self):
        return (self.endpoints.detach().clamp(0, 1) * 255).round().byte().cpu()

    def _encoded(self):
        version = self.endpoints._version
        if self._cache is not None and (bool(self.codec_frozen) or self._cache[0] == version):
            return self._cache
        source = self._source()
        # No-op clamp/device moves/load do not change the encoded texture. An
        # encoder-quality change alone is never a request to re-encode it.
        if self._cache is not None and torch.equal(source, self.encoded_source.cpu()):
            self._cache = (version, self._cache[1], self._cache[2])
            return self._cache
        if bool(self.codec_frozen):
            raise RuntimeError("Frozen ASTC proxy has no encoded state")
        rgba = np.full((*source.shape[:2], 4), 255, dtype=np.uint8)
        rgba[..., : source.shape[-1]] = source.numpy()
        data = self.codec.encode_rgba(rgba)
        decoded = self.codec.decode_bytes(data)[..., : source.shape[-1]].copy()
        self.bitstream = torch.tensor(list(data), dtype=torch.uint8, device=self.endpoints.device)
        self.encoded_source = source.to(self.endpoints.device)
        self._cache = (version, data, torch.from_numpy(decoded).float() / 255)
        return self._cache

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        self._encoded()
        super()._save_to_state_dict(destination, prefix, keep_vars)

    def _load_from_state_dict(self, state, prefix, *args, **kwargs):
        # Migration is confined to checkpoint loading.
        for name, default in [
            (
                "encoded_source",
                (state[prefix + "endpoints"].detach().clamp(0, 1) * 255).round().byte(),
            ),
            ("codec_frozen", self.codec_frozen),
        ]:
            state.setdefault(prefix + name, default)
        for name in ("bitstream", "encoded_source"):
            if prefix + name in state:
                setattr(
                    self, name, torch.empty_like(state[prefix + name], device=self.endpoints.device)
                )
        super()._load_from_state_dict(state, prefix, *args, **kwargs)
        if self.bitstream.numel():
            data = bytes(self.bitstream.cpu().tolist())
            if data[:4] != bytes.fromhex("13aba15c") or tuple(data[4:6]) != tuple(
                self.footprint.tolist()
            ):
                raise ValueError("ASTC checkpoint bitstream does not match footprint")
            decoded = self.codec.decode_bytes(data)[..., : self.endpoints.shape[-1]].copy()
            self._cache = (self.endpoints._version, data, torch.from_numpy(decoded).float() / 255)
        else:
            self._cache = None

    def decode_image(self):
        exact = self._encoded()[2].to(self.endpoints.device)
        if bool(self.codec_frozen):
            return exact
        return self.endpoints + (exact - self.endpoints).detach()

    def decode_cpu_image(self):
        return self._encoded()[2].clone()

    def export_astc(self):
        return self._encoded()[1]

    @torch.no_grad()
    def freeze(self):
        self._encoded()
        self.codec_frozen.fill_(True)
        self.requires_grad_(False)

    @torch.no_grad()
    def thaw(self):
        self.codec_frozen.fill_(False)
        self.requires_grad_(True)

    @torch.no_grad()
    def clamp_parameters(self):
        if not bool(self.codec_frozen):
            self.endpoints.clamp_(0, 1)

    @classmethod
    def from_state(cls, state, prefix, device, codec=None):
        block = "x".join(map(str, state[prefix + "footprint"].tolist()))
        codec = codec or ASTCCodec("tools/astcenc", "medium", block)
        if codec.astc_block != block:
            raise ValueError("ASTC checkpoint footprint does not match configuration")
        return cls(state[prefix + "endpoints"].to(device), codec)


class PackedASTCRoundtripGrid(nn.Module):
    """Several conventional ASTC surfaces with one logical feature image."""

    proxy_kind = "astc_roundtrip"

    def __init__(self, image, codec, groups):
        super().__init__()
        self.surfaces = nn.ModuleList(
            [ASTCRoundtripGrid(image[..., list(g)], codec) for g in groups]
        )
        self._image_shape = tuple(image.shape[:2])

    def decode_image(self):
        return torch.cat([s.decode_image() for s in self.surfaces], -1)

    def decode_cpu_image(self):
        return torch.cat([s.decode_cpu_image() for s in self.surfaces], -1)

    def export_surfaces(self):
        return tuple(s.export_astc() for s in self.surfaces)

    def freeze(self):
        for s in self.surfaces:
            s.freeze()

    def thaw(self):
        for s in self.surfaces:
            s.thaw()

    def clamp_parameters(self):
        for s in self.surfaces:
            s.clamp_parameters()
