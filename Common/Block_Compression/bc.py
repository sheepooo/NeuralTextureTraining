"""Traditional BC1 codec and DDS container (no neural-training dependency)."""

import math
import struct
import numpy as np


def dds_header(width, height):
    size = math.ceil(width / 4) * math.ceil(height / 4) * 8
    return (
        b"DDS "
        + struct.pack("<7I", 124, 0x81007, height, width, size, 0, 1)
        + bytes(44)
        + struct.pack("<8I", 32, 4, 0x31545844, 0, 0, 0, 0, 0)
        + struct.pack("<5I", 0x1000, 0, 0, 0, 0)
    )


def decode_blocks(data, width, height, rgba=False):
    bw, bh = math.ceil(width / 4), math.ceil(height / 4)
    if len(data) != bw * bh * 8:
        raise ValueError("BC1 payload length does not match image dimensions")
    words = np.frombuffer(data, dtype=np.dtype([("a", "<u2"), ("b", "<u2"), ("bits", "<u4")]))
    a, b = words["a"].astype(np.int32), words["b"].astype(np.int32)

    def rgb(values):
        r, g, blue = values >> 11, (values >> 5) & 63, values & 31
        return np.stack(((r << 3) | (r >> 2), (g << 2) | (g >> 4), (blue << 3) | (blue >> 2)), -1)

    x, y = rgb(a), rgb(b)
    palette = np.stack((x, y, (2 * x + y) // 3, (x + 2 * y) // 3), 1)
    punchthrough = a <= b
    palette[punchthrough, 2] = ((x + y) // 2)[punchthrough]
    palette[punchthrough, 3] = 0
    if rgba:
        alpha = np.full((*palette.shape[:2], 1), 255, dtype=np.int32)
        alpha[punchthrough, 3] = 0
        palette = np.concatenate((palette, alpha), -1)
    selectors = (words["bits"][:, None] >> (2 * np.arange(16, dtype=np.uint32))) & 3
    pixels = palette[np.arange(len(words))[:, None], selectors]
    channels = 4 if rgba else 3
    image = (
        pixels.reshape(bh, bw, 4, 4, channels)
        .transpose(0, 2, 1, 3, 4)
        .reshape(bh * 4, bw * 4, channels)
    )
    return image[:height, :width].astype(np.uint8)


class BC1Codec:
    """Conventional opaque/punchthrough BC1 encoder and byte-exact CPU decoder."""

    def encode_rgba(self, rgba):
        rgba = np.asarray(rgba, dtype=np.uint8)
        if rgba.ndim != 3 or rgba.shape[-1] != 4:
            raise ValueError("Expected uint8 HWC RGBA image")
        h, w, _ = rgba.shape
        bh, bw = math.ceil(h / 4), math.ceil(w / 4)
        padded = np.pad(rgba, ((0, bh * 4 - h), (0, bw * 4 - w), (0, 0)), mode="edge")
        pixels = padded.reshape(bh, 4, bw, 4, 4).transpose(0, 2, 1, 3, 4).reshape(-1, 16, 4)
        rgb = pixels[..., :3].astype(np.float32)
        opaque = pixels[..., 3] >= 128
        # Weighted PCA ignores transparent colors when fitting visible endpoints.
        count = opaque.sum(1, keepdims=True).clip(min=1)
        center = (rgb * opaque[..., None]).sum(1) / count
        delta = (rgb - center[:, None]) * opaque[..., None]
        covariance = delta.transpose(0, 2, 1) @ delta
        _, vectors = np.linalg.eigh(covariance)
        direction = vectors[:, :, -1]
        score = (rgb * direction[:, None]).sum(-1)
        high = np.where(opaque, score, -np.inf).argmax(1)
        low = np.where(opaque, score, np.inf).argmin(1)
        rows = np.arange(len(pixels))
        colors = np.stack((rgb[rows, high], rgb[rows, low]), 1)
        codes = np.rint(colors * np.array([31, 63, 31]) / 255).astype(np.int32)
        words = (codes[..., 0] << 11) | (codes[..., 1] << 5) | codes[..., 2]
        lo, hi = words.min(1), words.max(1)
        punchthrough = ~opaque.all(1)
        hi = np.where((hi == lo) & ~punchthrough, np.minimum(hi + 1, 65535), hi)
        lo = np.where((hi == lo) & ~punchthrough, np.maximum(lo - 1, 0), lo)
        a = np.where(punchthrough, lo, hi)
        b = np.where(punchthrough, hi, lo)

        def unpack(values):
            r, g, blue = values >> 11, (values >> 5) & 63, values & 31
            return np.stack(
                ((r << 3) | (r >> 2), (g << 2) | (g >> 4), (blue << 3) | (blue >> 2)), -1
            )

        x, y = unpack(a), unpack(b)
        palette = np.stack((x, y, (2 * x + y) // 3, (x + 2 * y) // 3), 1)
        palette[punchthrough, 2] = ((x + y) // 2)[punchthrough]
        palette[punchthrough, 3] = 0
        distance = ((rgb[:, :, None] - palette[:, None]) ** 2).sum(-1)
        distance[punchthrough, :, 3] = np.inf
        selectors = distance.argmin(-1).astype(np.uint32)
        selectors[~opaque] = 3
        bits = (
            (selectors << (2 * np.arange(16, dtype=np.uint32)))
            .sum(1, dtype=np.uint64)
            .astype(np.uint32)
        )
        blocks = np.empty(
            len(pixels), dtype=np.dtype([("a", "<u2"), ("b", "<u2"), ("bits", "<u4")])
        )
        blocks["a"], blocks["b"], blocks["bits"] = a, b, bits
        return dds_header(w, h) + blocks.tobytes()

    def decode_bytes(self, data):
        if len(data) < 128 or data[:4] != b"DDS " or data[84:88] != b"DXT1":
            raise ValueError("Expected BC1/DXT1 DDS")
        h, w = struct.unpack_from("<II", data, 12)
        return decode_blocks(data[128:], w, h, rgba=True)

    def roundtrip_rgba(self, rgba):
        return self.decode_bytes(self.encode_rgba(rgba))
