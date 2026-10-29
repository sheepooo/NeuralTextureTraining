"""Material raster reconstruction and image export shared by training and comparisons."""
import copy
import os
from typing import Optional, Tuple
import torch
import torchvision.transforms.functional as TF
from dataset import material_size, clean_base_image
from Core.Feature_Grid.sampling import resize_bilinear_repeat, sample_image
from Core.Material_Encoding.normal_encoding import decode_normal
from Core.Feature_Grid.compression import apply_latent_compression


def _ref_astc_hw_from_side(side: Optional[int], base_h: int, base_w: int) -> Tuple[int, int]:
    """Scale the longest material edge to side while preserving the GT aspect ratio."""
    if side is None:
        return base_h, base_w
    if side <= 0:
        raise ValueError(f"ref_astc_resolution must be a positive int (square edge), got {side}")
    scale = float(side) / float(max(base_h, base_w))
    return max(1, int(round(base_h * scale))), max(1, int(round(base_w * scale)))


def comparison_material_size(model, height: int, width: int, ref_astc_resolution=None):
    """Shared FNTC/ASTC raster, independent of latent resolution when specified."""
    edge = ref_astc_resolution or max(material_size(model, height, width))
    return _ref_astc_hw_from_side(edge, height, width)


@torch.no_grad()
def _render_gt_mip0(dataset, texture_height: int, texture_width: int) -> torch.Tensor:
    mip = 0
    mip_height = texture_height // (2 ** mip)
    mip_width = texture_width // (2 ** mip)
    gt_slice = dataset.mip_cache[mip]
    gt_canonical = dataset.expand_to_canonical(gt_slice.reshape(-1, gt_slice.shape[-1]))
    gt_canonical = gt_canonical.reshape(mip_height, mip_width, -1)
    return gt_canonical.permute(2, 0, 1)[None, ...]


@torch.no_grad()
def _render_model_mip0(model, dataset, texture_height: int, texture_width: int, num_mips: int, device: str,
                       superres_base_override: Optional[torch.Tensor] = None,
                       inference_tile: int = 512,
                       inference_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    """Render Mip0 in tiles, bounding decoder and feature-gradient temporaries."""
    mip = 0
    H = texture_height >> mip
    W = texture_width >> mip
    target_size = (H, W)
    H, W = inference_size if inference_size is not None else material_size(model, H, W)
    native_base = None
    if getattr(model, 'super_resolution_enable', False) and (H, W) != target_size:
        native_base = (clean_base_image(dataset, (H, W)) if superres_base_override is None
                       else superres_base_override)
        if tuple(native_base.shape[-2:]) != (H, W):
            raise ValueError('Render the compressed SR base at native material size before passing superres_base_override')
    mip_f = float(mip) / max(1, num_mips - 1)
    step = max(1, int(inference_tile) or max(H, W))
    hwc = None
    for h0 in range(0, H, step):
        h1 = min(h0 + step, H)
        for w0 in range(0, W, step):
            w1 = min(w0 + step, W)
            rr, cc = torch.meshgrid(torch.arange(h0, h1, device=device),
                                    torch.arange(w0, w1, device=device), indexing="ij")
            inp = torch.stack(((cc + 0.5) / W, (rr + 0.5) / H,
                               torch.full_like(rr, mip_f)), dim=-1).reshape(-1, 3)
            if getattr(model, "super_resolution_enable", False):
                if native_base is not None:
                    base = sample_image(native_base, inp[:, :2])
                elif superres_base_override is None:
                    base_data = dataset.superres_base_cache[mip][h0:h1, w0:w1]
                    base = dataset.expand_to_canonical(base_data.reshape(-1, dataset.num_channels)).float()
                else:
                    base_chw = superres_base_override[0] if superres_base_override.ndim == 4 else superres_base_override
                    base = base_chw[:, h0:h1, w0:w1].permute(1, 2, 0).reshape(-1, base_chw.shape[0]).float()
                y = (base + model(torch.cat([inp, base], dim=1)).float()).clamp(0.0, 1.0)
            else:
                y = model(inp).float()
            if hwc is None:
                hwc = torch.empty(H, W, y.shape[-1], device=device, dtype=torch.float32)
            hwc[h0:h1, w0:w1] = y.reshape(h1-h0, w1-w0, -1)
    if getattr(model, "direct_diffuse_enabled", False) and hasattr(model, "render_direct_diffuse_mip"):
        direct = model.render_direct_diffuse_mip(mip, H, W)
        hwc[:, :, :3] = direct.squeeze(0).permute(1, 2, 0).to(hwc.dtype)
    hwc = torch.nan_to_num(hwc, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    image = hwc.permute(2, 0, 1)[None, ...]
    if model.material_cnn is not None:
        image = model.material_cnn.tiled(image, inference_tile)
    return resize_bilinear_repeat(image, target_size) if (H, W) != target_size else image


@torch.no_grad()
def make_deployment_model(model):
    decoded = copy.deepcopy(model).eval()
    decoded.simulate_quantize()
    apply_latent_compression(decoded)
    decoded._decoder_adaptation_features = None
    decoded._compressed_material = True
    return decoded


def mip_plane_tile(
    h0: int, h1: int, w0: int, w1: int, plane_h: int, plane_w: int, mip_f: float, device: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """[N,3] batch and (row,col) indices; UV=(col+0.5)/W, (row+0.5)/H."""
    rr, cc = torch.meshgrid(
        torch.arange(h0, h1, device=device),
        torch.arange(w0, w1, device=device),
        indexing="ij",
    )
    z = torch.full_like(rr, mip_f, dtype=torch.float32)
    inp = torch.stack(((cc + 0.5) / plane_w, (rr + 0.5) / plane_h, z), dim=-1).reshape(-1, 3)
    return inp, rr.reshape(-1).long(), cc.reshape(-1).long()


@torch.no_grad()
def render_material_mip(model, dataset, mip, mip_height, mip_width, *, num_mips, device, inference_tile=512, ref_astc_resolution=None, superres_base_override=None) -> torch.Tensor:
    """Tiled full-plane forward (caps tcnn batch size; scatter by row/col to avoid layout bugs)."""
    if mip == 0:
        size = comparison_material_size(model, mip_height, mip_width, ref_astc_resolution)
        return _render_model_mip0(model, dataset, mip_height, mip_width, num_mips,
                                 device, superres_base_override=superres_base_override,
                                 inference_tile=inference_tile,
                                 inference_size=size)[0].permute(1, 2, 0)
    c = model.num_channels
    out = torch.empty(mip_height, mip_width, c, device=device, dtype=torch.float32)
    step = max(1, inference_tile or max(mip_height, mip_width))
    mip_f = 0.0 if num_mips <= 1 else float(mip) / float(num_mips - 1)
    for h0 in range(0, mip_height, step):
        h1 = min(h0 + step, mip_height)
        for w0 in range(0, mip_width, step):
            w1 = min(w0 + step, mip_width)
            inp, fr, fc = mip_plane_tile(h0, h1, w0, w1, mip_height, mip_width, mip_f, device)
            if getattr(model, 'super_resolution_enable', False):
                local_indices = torch.stack([fr, fc, torch.full_like(fr, mip)], dim=1)
                local_indices[:, :2] *= 2 ** mip
                base = dataset.get_superres_base(local_indices)
                base = dataset.expand_to_canonical(base).float()
                residual = model(torch.cat([inp, base], dim=1)).float()
                out[fr, fc, :] = (base + residual).clamp(0.0, 1.0)
            else:
                out[fr, fc, :] = model(inp).float()
    if getattr(model, "direct_diffuse_enabled", False) and "diffuse" in dataset.available_textures:
        s, e = dataset.canonical_channel_slices["diffuse"]
        direct = model.render_direct_diffuse_mip(mip, mip_height, mip_width)
        out[:, :, s:e] = direct.squeeze(0).permute(1, 2, 0).to(out.dtype)
    return out


def postprocess_material(image: torch.Tensor, vis_mode: str, normal_encoding='xyz') -> torch.Tensor:
    """Apply visualization post-processing based on vis_mode.
    Args:
        image: [C, H, W] tensor in [0, 1]
        vis_mode: 'srgb' | 'normal' | 'linear'
    """
    image = torch.clamp(image, 0.0, 1.0)
    if vis_mode == 'srgb':
        image = torch.pow(image, 1.0 / 2.2)
    elif vis_mode == 'normal':
        n = image * 2.0 - 1.0
        norm = torch.sqrt(torch.clamp((n ** 2).sum(dim=0, keepdim=True), min=1e-8))
        n = n / norm
        image = torch.clamp((n + 1.0) * 0.5, 0.0, 1.0)
    elif vis_mode == 'normal_encoded':
        image = decode_normal(image, normal_encoding)
    # 'linear' -> no transform
    return image


def canonical_gt_mip(dataset, mip):
    level = dataset.mip_cache[mip]
    return dataset.expand_to_canonical(level).permute(2, 0, 1)[None]


def save_material_visuals(pred, gt, output_root, filename, vis_configs, normal_encoding):
    image = torch.cat([pred, gt], dim=3)[0]
    for config in vis_configs:
        start, end = config['canonical_channel_slice']
        directory = os.path.join(output_root, config['display_name'])
        os.makedirs(directory, exist_ok=True)
        TF.to_pil_image(postprocess_material(image[start:end], config['vis_mode'], normal_encoding)).save(os.path.join(directory, filename))
