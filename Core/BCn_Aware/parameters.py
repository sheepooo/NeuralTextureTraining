"""BC1 algorithm-specific binary usage objective and diagnostics."""

import torch


def binary_regularization(trainer):
    losses = [
        p.binary_usage_loss() for p in trainer.model.astc_proxy if hasattr(p, "binary_usage_loss")
    ]
    if not losses:
        return 0.0
    loss = torch.stack(losses).mean()
    trainer.writer.add_scalar(
        "BC1/binary_usage_loss", loss.detach().item(), trainer.model.current_iter
    )
    if trainer.model.current_iter % 100 == 0:
        for index, proxy in enumerate(trainer.model.astc_proxy):
            if not hasattr(proxy, "usage_stats"):
                continue
            for name, value in proxy.usage_stats().items():
                trainer.writer.add_scalar(
                    f"BC1/grid{index}_{name}", value, trainer.model.current_iter
                )
    return trainer.configs.bcn_usage_weight * loss
