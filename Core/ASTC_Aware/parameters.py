"""ASTC-only native parameter updates, codec patches and offline refinement."""

import math
import torch
from Core.Feature_Grid.sampling import resize_bilinear_repeat
from dataset import material_size, prefiltered_material


class ASTCTrainingOperations:
    @torch.no_grad()
    def _project_astc_decoder_target(self):
        """Fit legal ASTC to a weighted linear-decoder target at grid texel centers.

        Unlike a normalized feature-gradient step, this proposal solves for the
        latent that minimizes material MSE at those centers before ASTC projection.
        Accept only the actual sampled material objective, keeping codec training
        on CUDA on all ordinary iterations.
        """
        from Core.ASTC_Aware.differentiable_proxy import ASTCProxyGrid, encode_seed
        from Core.Feature_Grid.compression import astc_codec_for
        from Core.Feature_Grid.feature_grid import feature_to_unorm

        self._validate_astc_projection()
        model = self.model
        grid = model.astc_proxy[0]
        spec = model.feature_grid_specs[0]
        channels = model.num_channels
        dim = 4 + 1 + channels + 1  # feature basis, mip, SR base, explicit bias
        basis = torch.zeros((5, dim), device=self.device)
        basis[1:, :4] = torch.eye(4, device=self.device)
        basis[:, -1] = 1
        values = model.network(basis).float()
        matrix = (values[1:] - values[:1]).T  # [channels,4]
        weights = torch.tensor(self.output_loss_weights, device=self.device)
        gram = matrix.T @ (weights[:, None] * matrix)
        ridge = 1e-5 * gram.diag().mean().clamp_min(1e-6)
        inverse = torch.linalg.inv(gram + torch.eye(4, device=self.device) * ridge)
        transform = (weights[:, None] * matrix) @ inverse
        resolution = spec.highest_level_slice()[2]
        # Solve at feature texel centers. Bilinear resizing samples the same UVs
        # from GT and the deployment SR base when their dimensions differ from
        # the grid (e.g. 2K textures with the default 1K feature grid).
        target_image = prefiltered_material(self.dataset, (resolution, resolution))
        base_image = self.astc_superres_base_mip0.permute(2, 0, 1)[None].float()
        if base_image.shape[-2:] != (resolution, resolution):
            base_image = resize_bilinear_repeat(base_image, (resolution, resolution))
        target = target_image[0].permute(1, 2, 0).reshape(-1, channels)
        base = base_image[0].permute(1, 2, 0).reshape(-1, channels)
        projected = []
        for begin in range(0, len(base), 65536):
            batch = base[begin : begin + 65536]
            inputs = torch.zeros((len(batch), dim), device=self.device)
            inputs[:, 5 : 5 + channels] = batch
            inputs[:, -1] = 1
            residual = (
                target[begin : begin + len(batch)]
                - batch
                - model.network(inputs).float()
                - model.decoder_output_bias.float()
            )
            projected.append(
                (residual @ transform).clamp(
                    spec.quant_min,
                    spec.quant_min + (spec.quant_step_count - 1) / spec.quant_step_count,
                )
            )
        latent = torch.cat(projected).reshape(resolution, resolution, 4)
        rgba = (feature_to_unorm(latent, spec) * 255).round().byte().cpu().numpy()
        proposal = ASTCProxyGrid.from_astc(
            encode_seed(astc_codec_for(self.model, 0), rgba), self.device
        )
        baseline = float(self._astc_proxy_objective())
        names = ("endpoints", "weights", "metadata", "partitions", "indices", "coefficients")
        saved = {name: getattr(grid, name).clone() for name in names}
        for name in names:
            getattr(grid, name).copy_(getattr(proposal, name))
        try:
            candidate = float(self._astc_proxy_objective())
        except Exception:
            for name in names:
                getattr(grid, name).copy_(saved[name])
            raise
        accepted = math.isfinite(candidate) and candidate < baseline
        if accepted:
            self._astc_block_optimizer.reset_rows(
                0, torch.arange(len(grid.endpoints), device=self.device)
            )
        else:
            for name in names:
                getattr(grid, name).copy_(saved[name])
        self.writer.add_scalar("ASTCProjection/accepted", int(accepted), model.current_iter)
        self.writer.add_scalar("ASTCProjection/baseline_loss", baseline, model.current_iter)
        self.writer.add_scalar("ASTCProjection/candidate_loss", candidate, model.current_iter)

    def _validate_astc_projection(self):
        model = self.model
        if (
            model.n_hidden_layers != 0
            or model.n_frequencies != 0
            or model.feature_gradient_count != 0
            or len(model.feature_grid_specs) != 1
            or not self.super_resolution_enable
        ):
            raise ValueError(
                "Material projection requires one grid and a linear SR decoder without PE/feature gradients; "
                "set astc_decoder_projection_interval: 0 for other configurations"
            )

    def refine_gradient_codec(self, steps, learning_rate=0.001, blocks_per_batch=64):
        """Fit legal codec symbols with the decoder fixed over complete neighborhoods.

        This explicit refinement pass is separate from frozen decoder adaptation.
        The output patch covers each selected 6x6 block and the gradient/bilinear
        footprint around it (10x10 for 3x3 inputs, 12x12 for 5x5). Selected rows are
        updated, and proposals must improve the actual material/PBR objective.
        """
        from Core.ASTC_Aware.differentiable_proxy import ASTCBlockAdam
        from Core.Feature_Grid.feature_gradients import gradient_directions

        model = self.model
        if (
            not self.configs.native_astc_only
            or self.astc_codec_backend != "astc_differentiable_proxy"
            or not self.mip0_only
            or len(model.astc_proxy) != 1
            or not model.feature_gradient_count
            or model._decoder_adaptation_features is not None
        ):
            raise ValueError(
                "Codec refinement requires a fitted gradient decoder and one Mip0 ASTC proxy"
            )
        if (
            steps < 1
            or blocks_per_batch < 1
            or not math.isfinite(learning_rate)
            or learning_rate <= 0
        ):
            raise ValueError("Refinement steps, block count and learning rate must be positive")
        grid = model.astc_proxy[0]
        height, width = grid._image_shape
        if (
            (height, width) != material_size(model, self.texture_height, self.texture_width)
            or self.texture_height % height
            or self.texture_width % width
        ):
            raise ValueError(
                "Codec refinement requires native material resolution and an integer GT scale"
            )
        scale_y, scale_x = self.texture_height // height, self.texture_width // width
        if self.super_resolution_enable and self.astc_superres_base_mip0 is None:
            raise ValueError("Codec refinement requires the deployment ASTC base")
        blocks_x, blocks_y = (width + 5) // 6, (height + 5) // 6
        optimizer = ASTCBlockAdam(
            model.astc_proxy,
            learning_rate,
            learning_rate,
            quantization_aware=True,
            packed_pair_search=True,
        )
        loss_weights = torch.tensor(self.output_loss_weights, device=self.device)
        halo = 1 + max(
            max(abs(dx), abs(dy)) for dx, dy in gradient_directions(model.feature_gradient_count)
        )
        if (scale_y, scale_x) != (1, 1):
            halo += 1
        ly, lx = torch.meshgrid(
            torch.arange(-halo * scale_y, (6 + halo) * scale_y, device=self.device),
            torch.arange(-halo * scale_x, (6 + halo) * scale_x, device=self.device),
            indexing="ij",
        )
        parameter_flags = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
        was_training, old_iteration = model.training, model.current_iter
        old_branch = model.astc_codec_branch_enabled
        old_batches = getattr(self, "_astc_proxy_batches", None)
        old_weights = getattr(self, "_astc_proxy_loss_weights", None)
        saved = (grid.endpoints.detach().clone(), grid.weights.detach().clone())
        saved_source = model.feature_grids[0].params.detach().clone()
        stats = {
            "steps": steps,
            "accepted_blocks": 0,
            "continuous_accepted": 0,
            "learning_rate": learning_rate,
            "blocks_per_batch": blocks_per_batch,
        }
        stats["loss_weights"] = list(self.output_loss_weights)
        try:
            model.train()
            for parameter, _ in parameter_flags:
                parameter.requires_grad_(False)
            grid.requires_grad_(True)
            for step in range(steps):
                bx = torch.randint(0, blocks_x, (blocks_per_batch,), device=self.device)
                by = torch.randint(0, blocks_y, (blocks_per_batch,), device=self.device)
                ids = (by * blocks_x + bx).unique()
                xs = (ids[:, None, None] % blocks_x * 6 * scale_x + lx) % self.texture_width
                ys = (ids[:, None, None] // blocks_x * 6 * scale_y + ly) % self.texture_height
                uv = torch.stack(
                    (
                        (xs.flatten() + 0.5) / self.texture_width,
                        (ys.flatten() + 0.5) / self.texture_height,
                    ),
                    1,
                )
                base = self._astc_base_at_uv(uv)
                target = self.dataset.expand_to_canonical(
                    self.dataset.mip_cache[0][ys.flatten(), xs.flatten()]
                ).to(torch.float16)
                inputs = torch.cat((uv, torch.zeros(len(uv), 1, device=self.device)), 1)
                if base is not None:
                    inputs = torch.cat((inputs, base), 1)
                model.current_iter = old_iteration + step + 1
                model._qat_param_cache.clear()
                optimizer.zero_grad()
                model.astc_codec_branch_enabled = True
                prediction = self._predict_material(inputs)
                loss = self._compute_astc_reconstruction_loss(
                    target, prediction, base, loss_weights
                )
                pbr = self._astc_pbr_loss(target, prediction, base)
                if pbr is not None:
                    loss = loss + self.pbr_loss_weight * pbr
                loss.backward()
                self._astc_proxy_batches = [(inputs, target, base, 1.0, True)]
                self._astc_proxy_loss_weights = loss_weights
                optimizer.step({0: ids}, objective=self._astc_proxy_objective)
                stats["accepted_blocks"] += optimizer.last_stats.get("accepted_blocks", 0)
                stats["continuous_accepted"] += optimizer.last_stats.get("continuous_accepted", 0)
            self._sync_astc_source()
            self._astc_block_optimizer = None
            return stats
        except Exception:
            with torch.no_grad():
                grid.endpoints.copy_(saved[0])
                grid.weights.copy_(saved[1])
                model.feature_grids[0].params.copy_(saved_source)
            raise
        finally:
            optimizer.zero_grad()
            for parameter, flag in parameter_flags:
                parameter.requires_grad_(flag)
            model.train(was_training)
            model.current_iter = old_iteration
            model.astc_codec_branch_enabled = old_branch
            model._qat_param_cache.clear()
            self._astc_proxy_batches = old_batches
            self._astc_proxy_loss_weights = old_weights
