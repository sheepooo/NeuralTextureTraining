"""Compression checkpoint migration and deployment metadata, isolated from models."""

import json
import torch
from Core.Feature_Grid.formats import GridCompression
from .compression import SourceGrid, astc_codec_for, grid_recipes, proxy_kind, plans_for
from .layout import surface_filename


def initialize_contract(model):
    data = json.dumps([r.as_config() for r in grid_recipes(model)], sort_keys=True).encode()
    model.register_buffer(
        "_grid_compression_contract",
        torch.tensor(list(data), dtype=torch.uint8, device=model.device),
    )


def validate_contract(model, state):
    expected = [r.as_config() for r in grid_recipes(model)]
    contract = state.get("_grid_compression_contract")
    if contract is not None:
        saved = [
            GridCompression.from_config(g).as_config()
            for g in json.loads(bytes(contract.cpu().tolist()).decode())
        ]
    else:
        # Pre-contract checkpoints stored a family marker and implicit ASTC6x6.
        legacy_bc = "_latent_format_id" in state
        mode = "rgba_binary" if "_bc1_layout_version" in state else "opaque_rgb"
        saved = []
        for i in range(len(expected)):
            prefix = f"astc_proxy.{i}."
            block = (
                "x".join(map(str, state[prefix + "footprint"].tolist()))
                if prefix + "footprint" in state
                else "6x6"
            )
            saved.append(GridCompression("bc1" if legacy_bc else "astc", block, mode).as_config())
    if saved != expected:
        raise ValueError("Checkpoint feature-grid compression recipes do not match configuration")
    state.pop("_latent_format_id", None)
    state.pop("_bc1_layout_version", None)
    state["_grid_compression_contract"] = model._grid_compression_contract.detach().clone()


def _state_kind(state, prefix):
    if prefix + "surfaces.0.endpoints" in state:
        return "astc_packed"
    if prefix + "endpoints" not in state:
        return "source"
    if prefix + "bc1_alpha_version" in state:
        return "bc1_rgba"
    if prefix + "footprint" in state:
        return "astc_roundtrip"
    if prefix + "metadata" in state:
        return "astc_native"
    return "bc1"


def restore_proxies(model, state):
    from Core.ASTC_Aware.roundtrip_proxy import ASTCRoundtripGrid, PackedASTCRoundtripGrid

    count = len(grid_recipes(model)) if any(k.startswith("astc_proxy.") for k in state) else 0
    if len(model.astc_proxy) == count and all(
        proxy_kind(p)
        == (
            "astc_roundtrip"
            if _state_kind(state, f"astc_proxy.{i}.") == "astc_packed"
            else _state_kind(state, f"astc_proxy.{i}.")
        )
        for i, p in enumerate(model.astc_proxy)
    ):
        return
    result = torch.nn.ModuleList()
    for i in range(count):
        prefix = f"astc_proxy.{i}."
        kind = _state_kind(state, prefix)
        if kind == "source":
            proxy = SourceGrid()
        elif kind == "astc_packed":
            plans = plans_for(model, i)
            images = [
                state[f"{prefix}surfaces.{j}.endpoints"].to(model.device) for j in range(len(plans))
            ]
            proxy = PackedASTCRoundtripGrid(
                torch.cat(images, -1), astc_codec_for(model, i), [p.channels for p in plans]
            )
        elif kind == "astc_roundtrip":
            proxy = ASTCRoundtripGrid.from_state(
                state, prefix, model.device, astc_codec_for(model, i)
            )
        else:
            if kind == "astc_native":
                from Core.ASTC_Aware.differentiable_proxy import ASTCProxyGrid as Proxy
            elif kind == "bc1_rgba":
                from Core.BCn_Aware.alpha_proxy import BC1AlphaProxyGrid as Proxy
            else:
                from Core.BCn_Aware.differentiable_proxy import BC1ProxyGrid as Proxy
            proxy = Proxy.from_state(state, prefix, model.device)
        result.append(proxy)
    model.astc_proxy = result


def deployment_arrays(model):
    manifest = []
    for i, (recipe, spec) in enumerate(zip(grid_recipes(model), model.feature_grid_specs)):
        plans = plans_for(model, i)
        manifest.append(
            dict(
                **recipe.as_config(),
                channels=spec.n_features_per_level,
                binary_channels=(
                    [j for j in range(spec.n_features_per_level) if j % 4 == 3]
                    if recipe.format == "bc1" and recipe.bc1_mode == "rgba_binary"
                    else []
                ),
                surfaces=[
                    dict(
                        file=surface_filename(i, j, recipe, len(plans)),
                        channels=list(p.channels),
                        block=list(p.block),
                        block_bytes=p.block_bytes,
                    )
                    for j, p in enumerate(plans)
                ],
            )
        )
    import numpy as np

    return dict(
        compression_schema_version=np.array(2, dtype=np.int32),
        feature_grid_compression=np.array(json.dumps(manifest)),
        latent_color_space=np.array("linear_unorm"),
    )
