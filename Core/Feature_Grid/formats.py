"""Immutable compression recipes for FNTC feature grids."""

import re
from dataclasses import dataclass

from Common.Block_Compression.astc import ASTC_2D_BLOCKS, normalize_astc_block


@dataclass(frozen=True)
class GridCompression:
    format: str = "astc"
    astc_block: str = "6x6"
    bc1_mode: str = "rgba_binary"

    def __post_init__(self):
        if self.format not in ("astc", "bc1"):
            raise ValueError(f"Unsupported feature grid format: {self.format}")
        if self.bc1_mode not in ("rgba_binary", "opaque_rgb"):
            raise ValueError(f"Unsupported BC1 mode: {self.bc1_mode}")
        object.__setattr__(
            self,
            "astc_block",
            normalize_astc_block(self.astc_block) if self.format == "astc" else "6x6",
        )
        if self.format == "astc":
            object.__setattr__(self, "bc1_mode", "rgba_binary")

    @classmethod
    def from_config(
        cls, cfg, default_format="astc", default_block="6x6", default_bc1_mode="rgba_binary"
    ):
        token = str(cfg.get("format", default_format)).strip().lower()
        match = re.fullmatch(r"astc(\d+[xX]\d+)", token)
        if match:
            block = normalize_astc_block(match.group(1))
            if "bc1_mode" in cfg:
                raise ValueError("BC1 mode cannot be set on an ASTC grid")
            if "astc_block" in cfg and normalize_astc_block(cfg["astc_block"]) != block:
                raise ValueError("format and legacy astc_block disagree")
            return cls("astc", block)
        if token == "bc1":
            if "astc_block" in cfg:
                raise ValueError("astc_block only applies to ASTC")
            # BC1 means the native RGBA layout: RGB plus a binary alpha.
            # Opaque RGB is accepted only for historical checkpoint migration.
            return cls(
                "bc1",
                default_block,
                cfg.get("bc1_mode", "rgba_binary" if "format" in cfg else default_bc1_mode),
            )
        if token == "astc":
            if "bc1_mode" in cfg:
                raise ValueError("bc1_mode only applies to BC1")
            return cls("astc", cfg.get("astc_block", default_block))
        raise ValueError(f"Unsupported feature grid format {token!r}; use astc6x6, astc4x4 or BC1")

    @property
    def tag(self):
        return f"astc_{self.astc_block}" if self.format == "astc" else "bc1"

    @property
    def block(self):
        return tuple(map(int, self.astc_block.split("x"))) if self.format == "astc" else (4, 4)

    @property
    def channels_per_surface(self):
        return 3 if self.format == "bc1" and self.bc1_mode == "opaque_rgb" else 4

    def as_config(self):
        if self.format == "astc":
            return {"format": "astc" + self.astc_block}
        if self.bc1_mode == "opaque_rgb":
            # Legacy checkpoint recipe, not a setting needed for normal BC1.
            return {"format": "BC1", "bc1_mode": "opaque_rgb"}
        return {"format": "BC1"}
