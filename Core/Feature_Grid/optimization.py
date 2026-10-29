"""One per-grid training lifecycle for native ASTC, codec STE and BCn."""

import torch
from Core.Feature_Grid.compression import (
    initialize_grid_proxy,
    grid_recipes,
    proxy_kind,
    SourceGrid,
)


class GridCompressionOptimizer:
    def __init__(self, trainer):
        self.trainer = trainer
        model = trainer.model
        recipes = grid_recipes(model)
        if not len(model.astc_proxy):
            model.astc_proxy = torch.nn.ModuleList([SourceGrid() for _ in recipes])
        self.optimizers = []
        for i, recipe in enumerate(recipes):
            enabled = (
                trainer.configs.bcn_aware_enable
                if recipe.format == "bc1"
                else trainer.configs.astc_aware_enable
            )
            proxy = model.astc_proxy[i]
            if not enabled:
                # Source features continue training. Old compressed state was
                # already synchronized to the ordinary grid at each update.
                model.astc_proxy[i] = SourceGrid()
                self.optimizers.append(None)
                continue
            if proxy_kind(proxy) == "source":
                proxy = initialize_grid_proxy(model, i, trainer.astc_codec)
                model.astc_proxy[i] = proxy
            if proxy_kind(proxy) == "astc_roundtrip":
                proxy.thaw()
            else:
                proxy.requires_grad_(True)
            if proxy_kind(proxy) == "astc_native":
                from Core.ASTC_Aware.differentiable_proxy import ASTCBlockAdam

                optimizer = ASTCBlockAdam(
                    torch.nn.ModuleList([proxy]),
                    trainer.configs.astc_learning_rate,
                    trainer.configs.astc_learning_rate,
                    fused=True,
                    quantization_aware=True,
                    packed_pair_search=True,
                )
            else:
                lr = (
                    trainer.configs.bcn_learning_rate
                    if recipe.format == "bc1"
                    else trainer.configs.astc_learning_rate
                )
                optimizer = torch.optim.Adam(proxy.parameters(), lr=lr)
            self.optimizers.append(optimizer)
        self.iteration = 0

    def prepare(self):
        trainer = self.trainer
        progress = min(1.0, max(0.0, trainer.model.current_iter / max(1, trainer.max_iter - 1)))
        for proxy in trainer.model.astc_proxy:
            if proxy_kind(proxy) == "bc1_rgba":
                proxy.temperature.fill_(0.25**progress)

    def zero_grad(self):
        for opt in self.optimizers:
            if opt is not None:
                opt.zero_grad()

    def step(self):
        for index, opt in enumerate(self.optimizers):
            if opt is None:
                continue
            if isinstance(opt, torch.optim.Optimizer):
                opt.step()
            elif self.iteration % 100 == 0 and hasattr(self.trainer, "_astc_proxy_batches"):
                proxy = self.trainer.model.astc_proxy[index]
                touched = torch.where(
                    (proxy.endpoints.grad.abs().sum(1) + proxy.weights.grad.abs().sum(1)) > 0
                )[0]
                opt.step({0: touched}, objective=self.trainer._astc_proxy_objective)
            else:
                opt.step_full_grid()
        self.trainer.model._qat_param_cache.clear()
        self.iteration += 1
        self.trainer._astc_codec_pass_count = self.iteration
        interval = self.trainer.configs.astc_decoder_projection_interval
        if interval > 0 and self.iteration % interval == 0:
            self.trainer._project_astc_decoder_target()

    def reset_rows(self, index, rows):
        self.optimizers[index].reset_rows(0, rows)
