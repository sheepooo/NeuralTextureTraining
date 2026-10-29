"""FNTC grid compression boundary: deployment, sampling and shared surface plans."""

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch import nn

from Core.Feature_Grid.formats import GridCompression
from Common.Block_Compression.astc import ASTCCodec
from Common.Block_Compression.bc import dds_header
from .layout import surface_filename
from Core.Feature_Grid.feature_grid import (
    feature_to_unorm,
    unorm_to_feature,
    quantize_feature_tensor,
    quantize_feature_ste,
)


class SourceGrid(nn.Module):
    """Explicit marker for an ordinary, non-aware feature source."""

    proxy_kind = "source"


def grid_recipes(model, default_block="6x6"):
    # Lightweight legacy experiment objects can still supply a global recipe;
    # maintained models/configs always own one immutable tuple of recipes.
    if hasattr(model, "grid_compressions"):
        return model.grid_compressions
    return tuple(
        GridCompression(
            getattr(model, "latent_format", "astc"),
            getattr(model, "astc_block", default_block),
            getattr(model, "bc1_mode", "opaque_rgb"),
        )
        for _ in model.feature_grid_specs
    )


def compressed_method(model, block="6x6"):
    tags = {r.tag for r in grid_recipes(model, block)}
    return "fntc_" + next(iter(tags)) if len(tags) == 1 else "fntc_mixed"


def plans_for(model, index):
    from .layout import surface_plan

    return surface_plan(
        model.feature_grid_specs[index],
        grid_recipes(model)[index],
        getattr(model, "direct_diffuse_enabled", False) and index == 0,
    )


def latent_payload_bytes(model, astc_block="6x6"):
    return sum(
        p.payload_bytes(spec.max_resolution, spec.max_resolution)
        for i, spec in enumerate(model.feature_grid_specs)
        for p in plans_for(model, i)
    )


def grid_image(model, index):
    spec = model.feature_grid_specs[index]
    offset, count, res = spec.highest_level_slice()
    source = model._get_grid_params_tensor(model.feature_grids[index]).detach()[
        offset : offset + count
    ]
    return feature_to_unorm(quantize_feature_tensor(source, spec), spec).reshape(
        res, res, spec.n_features_per_level
    )


def astc_codec_for(model, index, fallback=None):
    recipe = grid_recipes(model)[index]
    return ASTCCodec(
        getattr(model, "astcenc_path", getattr(fallback, "astcenc_path", "tools/astcenc")),
        getattr(model, "astcenc_quality", getattr(fallback, "astcenc_quality", "medium")),
        recipe.astc_block,
    )


def proxy_kind(proxy):
    return getattr(proxy, "proxy_kind", "source")


def initialize_grid_proxy(model, index, fallback=None):
    recipe = grid_recipes(model)[index]
    image = grid_image(model, index)
    if recipe.format == "bc1":
        if recipe.bc1_mode == "rgba_binary":
            from Core.BCn_Aware.alpha_proxy import BC1AlphaProxyGrid as Proxy
        else:
            from Core.BCn_Aware.differentiable_proxy import BC1ProxyGrid as Proxy
        return Proxy.from_image(image)
    codec = astc_codec_for(model, index, fallback)
    plans = plans_for(model, index)
    if (
        recipe.astc_block == "6x6"
        and image.shape[-1] == 4
        and len(plans) == 1
        and getattr(model, "astc_codec_backend", "") == "astc_differentiable_proxy"
    ):
        from Core.ASTC_Aware.differentiable_proxy import ASTCProxyGrid, encode_seed

        return ASTCProxyGrid.from_astc(
            encode_seed(codec, (image * 255).round().byte().cpu().numpy()), model.device
        )
    from Core.ASTC_Aware.roundtrip_proxy import ASTCRoundtripGrid, PackedASTCRoundtripGrid

    if len(plans) > 1:
        return PackedASTCRoundtripGrid(image, codec, [p.channels for p in plans])
    return ASTCRoundtripGrid(image, codec)


@dataclass(frozen=True)
class EncodedGrid:
    streams: tuple[bytes, ...]
    image: torch.Tensor


def encode_grid(model, index):
    recipe = grid_recipes(model)[index]
    proxies = getattr(model, "astc_proxy", ())
    proxy = proxies[index] if len(proxies) > index else SourceGrid()
    if proxy_kind(proxy) != "source":
        if recipe.format == "bc1":
            streams = tuple(proxy.export_blocks())
        elif hasattr(proxy, "export_surfaces"):
            streams = proxy.export_surfaces()
        else:
            streams = (proxy.export_astc(),)
        return EncodedGrid(streams, proxy.decode_cpu_image())
    if recipe.format == "bc1":
        proxy = initialize_grid_proxy(model, index)
        return EncodedGrid(tuple(proxy.export_blocks()), proxy.decode_cpu_image())
    image = grid_image(model, index)
    codec = astc_codec_for(model, index)
    streams = []
    decoded = []
    for plan in plans_for(model, index):
        rgba = np.full((*image.shape[:2], 4), 255, dtype=np.uint8)
        rgba[..., : len(plan.channels)] = (
            (image[..., list(plan.channels)].cpu().numpy() * 255).round().astype(np.uint8)
        )
        data = codec.encode_rgba(rgba)
        streams.append(data)
        restored = codec.decode_bytes(data)[..., : len(plan.channels)].copy()
        decoded.append(torch.from_numpy(restored).float() / 255)
    return EncodedGrid(tuple(streams), torch.cat(decoded, -1))


@torch.no_grad()
def apply_latent_compression(model):
    for i, (grid, spec) in enumerate(zip(model.feature_grids, model.feature_grid_specs)):
        image = encode_grid(model, i).image.to(grid.params.device)
        offset, count, _ = spec.highest_level_slice()
        grid.params[offset : offset + count].copy_(
            quantize_feature_tensor(unorm_to_feature(image.reshape(-1), spec), spec)
        )


def export_latents(model, directory):
    for i, recipe in enumerate(grid_recipes(model)):
        encoded = encode_grid(model, i)
        plans = plans_for(model, i)
        if len(encoded.streams) != len(plans):
            raise RuntimeError("Codec output does not match surface plan")
        for surface, (stream, plan) in enumerate(zip(encoded.streams, plans)):
            filename = surface_filename(i, surface, recipe, len(plans))
            if recipe.format == "bc1":
                res = model.feature_grid_specs[i].max_resolution
                data = dds_header(res, res) + stream
                payload = len(stream)
            else:
                data = stream
                payload = len(stream) - 16
            expected = plan.payload_bytes(
                model.feature_grid_specs[i].max_resolution,
                model.feature_grid_specs[i].max_resolution,
            )
            if payload != expected:
                raise RuntimeError("Encoded payload does not match surface plan")
            Path(directory, filename).write_bytes(data)


def clamp_proxies(model):
    for proxy in model.astc_proxy:
        if proxy_kind(proxy) != "source" and any(p.requires_grad for p in proxy.parameters()):
            proxy.clamp_parameters()


def freeze_proxies(model):
    if not len(model.astc_proxy):
        model.astc_proxy = torch.nn.ModuleList([SourceGrid() for _ in grid_recipes(model)])
    for i, proxy in enumerate(model.astc_proxy):
        if proxy_kind(proxy) == "source":
            proxy = initialize_grid_proxy(model, i)
            model.astc_proxy[i] = proxy
        if proxy_kind(proxy) == "astc_roundtrip":
            proxy.freeze()
        else:
            proxy.requires_grad_(False)


def sample_proxy_grid(model, grid_idx, feature_grid):
    proxy = model.astc_proxy[grid_idx]
    spec = model.feature_grid_specs[grid_idx]
    if proxy_kind(proxy) == "source":
        return quantize_feature_ste(model._get_grid_params_tensor(feature_grid), spec)
    version = tuple(p._version for p in proxy.parameters())
    key = (grid_idx, "codec")
    cached = model._qat_param_cache.get(key)
    if cached is not None and cached[0] == version:
        return cached[1]
    offset, count, _ = spec.highest_level_slice()
    decoded = quantize_feature_ste(
        proxy.decode_image().reshape(-1) * ((spec.quant_step_count - 1) / spec.quant_step_count)
        + spec.quant_min,
        spec,
    )
    source = model._get_grid_params_tensor(feature_grid).detach()
    active = torch.cat((source[:offset], decoded, source[offset + count :]))
    model._qat_param_cache[key] = (version, active)
    return active
