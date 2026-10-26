"""Material reconstruction objectives, independent of training and reporting."""
import torch
from typing import List
from Core.Material_Encoding.normal_encoding import normal_angular_loss, normal_vectors


class MaterialLoss:
    def __init__(self, dataset, normal_encoding):
        self.dataset = dataset
        self.normal_encoding = normal_encoding
        self._loss_group_indices = {}

    def _group_indices(self, textures: List[str], device: torch.device) -> torch.Tensor:
        key = (tuple(textures), device)
        if key in self._loss_group_indices:
            return self._loss_group_indices[key]
        idx = []
        for tex in textures:
            if tex not in self.dataset.available_textures:
                continue
            s, e = self.dataset.canonical_channel_slices[tex]
            idx.extend(range(s, e))
        indices = torch.tensor(idx, device=device, dtype=torch.long)
        self._loss_group_indices[key] = indices
        return indices

    def pbr(self, gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """Differentiable shaded-RGB loss for PBR-sensitive material channels."""
        slices = self.dataset.canonical_channel_slices
        def get_pair(name, channels=1, default=0.0):
            if name not in self.dataset.available_textures:
                shape = (gt.shape[0], channels)
                value = torch.full(shape, default, device=gt.device, dtype=gt.dtype)
                return value, value
            s, e = slices[name]
            return (torch.nan_to_num(gt[:, s:e].float(), nan=default, posinf=default, neginf=default).clamp(0, 1),
                    torch.nan_to_num(pred[:, s:e].float(), nan=default, posinf=default, neginf=default).clamp(0, 1))

        gt_diff, pr_diff = get_pair("diffuse", 3, 0.5)
        gt_norm, pr_norm = get_pair("normal", getattr(self.dataset, "normal_encoding", "xyz") == "xyz" and 3 or 2, 0.5)
        gt_rough, pr_rough = get_pair("roughness", 1, 0.5)
        gt_metal, pr_metal = get_pair("metallic", 1, 0.0)
        gt_spec, pr_spec = get_pair("specular", 1, 0.5)

        gt_n = normal_vectors(gt_norm, self.normal_encoding)
        pr_n = normal_vectors(pr_norm, self.normal_encoding)
        # Use a fixed tangent-space light with positive Z so a flat normal map
        # receives direct light and all PBR channels get useful gradients.
        lighting = getattr(self, "_pbr_lighting", None)
        if lighting is None or lighting[0].device != pred.device or lighting[0].dtype != pred.dtype:
            light = torch.tensor([-0.35, 0.45, 0.82], device=pred.device, dtype=pred.dtype)
            light = light / torch.clamp(torch.linalg.vector_norm(light), min=1e-6)
            view = torch.tensor([0.0, 0.0, 1.0], device=pred.device, dtype=pred.dtype)
            half = (light + view) / torch.clamp(torch.linalg.vector_norm(light + view), min=1e-6)
            self._pbr_lighting = (light, view, half)
        light, view, half = self._pbr_lighting
        def shade(diff, normal, rough, metal, spec):
            diff = torch.clamp(diff, 0, 1)
            ndotl = torch.clamp((normal * light[None]).sum(1, keepdim=True), 0, 1)
            ndoth = torch.clamp((normal * half[None]).sum(1, keepdim=True), 0, 1)
            vdoth = torch.clamp((view[None] * half[None]).sum(1, keepdim=True), 0, 1)
            # Avoid the singular GGX lobe at roughness=0.045 during training.
            rough = torch.clamp(rough, 0.12, 1.0)
            alpha = rough * rough
            d_den = (ndoth * ndoth * (alpha * alpha - 1) + 1) ** 2
            d = alpha * alpha / (torch.pi * torch.clamp(d_den, min=0.04))
            d = torch.clamp(d, max=8.0)
            f0 = (0.02 + 0.14 * spec) * (1 - metal) + diff * metal
            fresnel = torch.clamp(f0 + (1 - f0) * (1 - vdoth) ** 5, 0.0, 1.0)
            shaded = torch.clamp(diff * (1 - metal) * (1 - fresnel) * ndotl / torch.pi
                                 + d * fresnel * ndotl * 0.25 + diff * 0.03, 0, 1)
            return torch.nan_to_num(shaded, nan=0.0, posinf=1.0, neginf=0.0)
        pred_shaded = shade(pr_diff, pr_n, pr_rough, pr_metal, pr_spec)
        gt_shaded = shade(gt_diff, gt_n, gt_rough, gt_metal, gt_spec).detach()
        return torch.nan_to_num(torch.nn.functional.smooth_l1_loss(pred_shaded, gt_shaded),
                                nan=0.0, posinf=1.0, neginf=0.0)

    def reconstruction(
        self,
        gt_texture: torch.Tensor,
        predict_texture: torch.Tensor,
        loss_weights: torch.Tensor,
    ):
        total_loss = torch.tensor(0.0, device=gt_texture.device, dtype=torch.float32)
        stats = {}

        for group, textures in self.dataset.texture_packing.items():
            idx = self._group_indices(textures, gt_texture.device)
            if idx.numel() == 0:
                stats[group] = None
                continue

            pred = torch.index_select(predict_texture.float(), dim=1, index=idx)
            gt = torch.index_select(gt_texture.float(), dim=1, index=idx)
            weights = loss_weights.index_select(0, idx).float()
            weight = weights.mean()

            if group == "normal":
                raw_loss = normal_angular_loss(pred, gt, self.normal_encoding)
                weighted_loss = raw_loss * weights.sum()
            else:
                mse = (gt - pred).pow(2).mean(dim=0)
                raw_loss = mse.mean()
                weighted_loss = (mse * weights).sum()

            total_loss = total_loss + weighted_loss
            stats[group] = {
                "loss": float(weighted_loss.detach().item()),
                "raw_loss": float(raw_loss.detach().item()),
                "weight": float(weight.detach().item()),
            }

        return total_loss, stats

    def residual(self, target, prediction, weights, statistics=False):
        mse = (target.float() - prediction.float()).square().mean(dim=0)
        loss = (mse * weights.float()).sum()
        stats = {}
        if statistics:
            groups = {}
            for group, textures in self.dataset.texture_packing.items():
                idx = self._group_indices(textures, target.device)
                if idx.numel():
                    groups[group] = (mse.index_select(0, idx) * weights.index_select(0, idx).float()).sum().detach()
            values = torch.stack([loss.detach(), *groups.values()]).tolist()
            stats = dict(loss=values[0], **{name: {'loss': value} for name, value in zip(groups, values[1:])})
        return loss, stats
