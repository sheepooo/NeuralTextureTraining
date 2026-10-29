"""Multi-stage training for ASTC: representation learning, decoder fitting and layout refitting.

Run as a module or through train.py with the same CLI/YAML options:
    python -m Core.Multi_Stage_Training.multi_stage_training --data_dir data/test
"""

import copy
import json
import os
import math
import time

import torch

from configs import Config, get_args
from model import TCNNModel
from train import Trainer
from .layout import refit_layout


def fit_astc_decoder(trainer, steps: int = 1000, lr_multiplier: float = 0.5,
                     early_stop: bool = False, final_report: bool = True):
    """Keep source latents intact while fitting their fixed ASTC-decoded inputs."""
    from Core.Feature_Grid.compression import apply_latent_compression

    if steps < 1 or not math.isfinite(lr_multiplier) or lr_multiplier <= 0:
        raise ValueError("steps and lr_multiplier must be positive")
    model = trainer.model
    if model.current_iter < 0 or model.current_iter >= trainer.max_iter:
        raise ValueError("Decoder adaptation requires a completed baseline iteration")
    if not trainer.mip0_only or not model.quantize or model.direct_diffuse_enabled:
        raise ValueError("ASTC decoder adaptation requires quantized Mip0 without direct diffuse")
    if model._decoder_adaptation_features is not None:
        raise RuntimeError("Decoder adaptation is already active")
    if trainer.super_resolution_enable and trainer.astc_superres_base_mip0 is None:
        raise ValueError("A deployment ASTC base is required for super-resolution adaptation")

    from Core.Feature_Grid.compression import freeze_proxies
    freeze_proxies(model)
    source = [grid.params.detach().clone() for grid in model.feature_grids]
    decoded = copy.deepcopy(model)
    decoded.simulate_quantize()
    apply_latent_compression(decoded)
    fixed = [grid.params.detach().clone() for grid in decoded.feature_grids]
    del decoded
    fields = ("trained_iter", "max_iter", "eval_interval", "save_interval", "early_stop",
              "_early_stop_phase_start_iter",
              "enable_astc_compare", "enable_pbr_compare", "_phase_control")
    previous = {name: getattr(trainer, name) for name in fields}
    old_scheduler = model.scheduler
    old_codec = trainer.configs.codec_training_enable
    old_base = trainer.dataset.superres_base_cache[0] if trainer.super_resolution_enable else None
    old_requires_grad = [grid.params.requires_grad for grid in model.feature_grids]
    old_lrs = [group["lr"] for group in model.optimizer.param_groups]
    was_training = model.training
    started = time.perf_counter()
    try:
        model._decoder_adaptation_features = fixed
        for grid in model.feature_grids:
            grid.params.requires_grad_(False)
        if trainer.super_resolution_enable and tuple(trainer.astc_superres_base_mip0.shape[:2]) == tuple(trainer.dataset.superres_base_cache[0].shape[:2]):
            slices = trainer.dataset.canonical_channel_slices
            trainer.dataset.superres_base_cache[0] = torch.cat([
                trainer.astc_superres_base_mip0[..., slices[name][0]:slices[name][1]]
                for name in trainer.dataset.available_textures
            ], dim=-1)
        trainer.configs.codec_training_enable = False
        model.optimizer.param_groups[0]["lr"] *= lr_multiplier
        # Validation-window LR reductions replace the per-batch scheduler.
        model.scheduler = None
        trainer.trained_iter = model.current_iter + 1
        trainer.max_iter = trainer.trained_iter + steps
        trainer._reset_training_phase("decoder", trainer.trained_iter)
        trainer.early_stop = early_stop
        model.train()
        trainer.train(final_report=final_report)
        torch.cuda.synchronize()
        for grid, original in zip(model.feature_grids, source):
            if not torch.equal(grid.params.detach(), original):
                raise RuntimeError("Frozen source latent changed during adaptation")
        for parameter in model.network.parameters():
            if not torch.isfinite(parameter).all():
                raise RuntimeError("Nonfinite decoder parameters after adaptation")
        completed_steps = model.current_iter - trainer.trained_iter + 1
        return {"steps": completed_steps, "max_steps": steps,
                "early_stopped": completed_steps < steps,
                "lr_multiplier": lr_multiplier,
                "seconds": time.perf_counter() - started, "frozen_latent_verified": True,
                "validation": trainer.last_training_stats}
    finally:
        model._decoder_adaptation_features = None
        model.scheduler = old_scheduler
        model.train(was_training)
        trainer.configs.codec_training_enable = old_codec
        if trainer.super_resolution_enable:
            trainer.dataset.superres_base_cache[0] = old_base
        for grid, requires_grad in zip(model.feature_grids, old_requires_grad):
            grid.params.requires_grad_(requires_grad)
        for group, lr in zip(model.optimizer.param_groups, old_lrs):
            group["lr"] = lr
        for name, value in previous.items():
            setattr(trainer, name, value)
        model._qat_param_cache.clear()
        model._qat_param_cache_iter = -1


def _record_stage(trainer, stages, name, start_iteration, details, status='completed'):
    """Save one named stage using a common checkpoint and metric schema."""
    model = trainer.model
    iteration = int(model.current_iter)
    trainer._metrics_phase = name
    model.save(iteration, trainer.model_path)
    astc = trainer.run_astc_comparison(curr_iter=iteration)
    pbr = (trainer.run_pbr_comparison(curr_iter=iteration, return_metrics=True)[1]
           if trainer.enable_pbr_compare else None)
    record = {
        "name": name, "status": status,
        "start_iteration": start_iteration, "end_iteration": iteration,
        "checkpoint": os.path.join(trainer.model_path, f"train_result_{iteration}", "model.pth"),
        "model": {"hidden_layers": model.n_hidden_layers, "neurons": model.n_neurons,
                  "pe_frequencies": model.n_frequencies,
                  "feature_gradient_count": model.feature_gradient_count,
                  "material_cnn_enable": model.material_cnn is not None},
        "metrics": {"astc": astc, "pbr": pbr}, "details": details,
        "metric_reports": {
            "astc": os.path.join(trainer.media_path, f'Compare_ASTC{trainer.astc_block.lower()}',
                                  f'iter_{iteration:08d}', 'metrics.json'),
            "pbr": os.path.join(trainer.media_path, 'Compare_PBRScene', f'iter_{iteration:08d}', 'metrics.json')
                   if trainer.enable_pbr_compare else None,
        },
    }
    stages.append(record)
    _write_stage_report(trainer, stages)
    return record


def _write_stage_report(trainer, stages, completed=False):
    latest = next((stage for stage in reversed(stages) if 'checkpoint' in stage), None)
    report = dict(status='completed' if completed else 'running', stages=stages)
    if latest is not None:
        report.update(final_iteration=latest['end_iteration'], final_checkpoint=latest['checkpoint'],
                      final_metrics=latest['metrics'])
    path = os.path.join(trainer.save_path, 'multi_stage_training.json')
    with open(path + '.tmp', 'w', encoding='utf-8') as output:
        json.dump(report, output, indent=2)
    os.replace(path + '.tmp', path)
    return report


def _finish_material_cnn(trainer, stages, steps, resume_state=None):
    print('[Multi Stage Training] material_cnn', flush=True)
    start = trainer.model.current_iter + 1
    details = trainer.fit_material_cnn(steps, resume_state=resume_state)
    progress_path = details.pop('progress_checkpoint')
    trainer.model.current_iter += details['steps']
    trainer.configs.material_cnn_enable = trainer.model.material_cnn is not None
    trainer._write_resolved_config(trainer.configs)
    _record_stage(trainer, stages, 'material_cnn', start, details,
                  status='completed' if details['accepted'] else 'rejected')
    trainer.save_compressed_materials(trainer.media_path, trainer.model.current_iter)
    # Completion means all final artifacts exist. Keep recovery data on failure.
    report = _write_stage_report(trainer, stages, completed=True)
    os.remove(progress_path)
    return report


def _resume_material_cnn(params, target, path):
    """Resume only the interrupted CNN stage from its frozen source and optimizer."""
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state['version'] != 2:
        raise ValueError('Unsupported material CNN training checkpoint version')
    if not target.material_cnn_enable:
        raise ValueError('Resuming the material CNN stage requires material_cnn_enable')
    config = copy.deepcopy(target)
    config.load_dir, config.load_iter = None, 0
    config.material_cnn_enable = False
    config.astc_aware_enable = config.codec_training_enable = False
    config.astc_decoder_projection_interval = 0
    trainer = Trainer(params, config_override=config, deployment_config=target)
    try:
        trainer.model.load_state_dict(state['source_model'])
        trainer.model.current_iter = state['source_iteration']
        with open(os.path.join(target.load_dir, 'multi_stage_training.json'), encoding='utf-8') as source:
            stages = [stage for stage in json.load(source)['stages'] if stage['name'] != 'material_cnn']
        _write_stage_report(trainer, stages)
        return _finish_material_cnn(trainer, stages, target.material_cnn_steps, resume_state=state)
    finally:
        trainer.writer.close()


def run_multi_stage_training(params, config=None):
    """Run representation, decoder and optional layout stages without numeric stage names."""
    target = config if config is not None else Config(params)
    if target.load_dir:
        progress = os.path.join(target.load_dir, 'models', 'material_cnn_training.pth')
        if os.path.isfile(progress):
            return _resume_material_cnn(params, target, progress)
    if not target.mip0_only or target.direct_diffuse_infer_mode != "disable":
        raise ValueError("Multi-stage ASTC training requires Mip0 and direct diffuse disabled")
    if target.n_hidden_layers < 0 or target.multi_stage_training_decoder_steps < 1:
        raise ValueError("Decoder stage requires nonnegative hidden layers and positive fitting steps")
    if target.early_stop and (target.early_stop_interval <= 0
                              or target.early_stop_interval % target.eval_interval != 0):
        raise ValueError("early_stop_interval must be positive and a multiple of eval_interval")

    representation_config = copy.deepcopy(target)
    representation_config.n_hidden_layers = 0
    representation_config.n_frequencies = 0
    representation_config.feature_gradient_count = 0
    representation_config.material_cnn_enable = False
    trainer = Trainer(params, config_override=representation_config, deployment_config=target)
    stages = []
    try:
        print("[Multi Stage Training] representation: affine decoder, no PE or feature differences", flush=True)
        start = trainer.trained_iter
        trainer.train(final_report=False)
        representation = _record_stage(trainer, stages, "representation", start,
                                       {"validation": trainer.last_training_stats})
        os.replace(os.path.join(trainer.save_path, "resolved_config.json"),
                   os.path.join(trainer.save_path, "representation_config.json"))

        source = trainer.model
        target.num_mips = representation_config.num_mips
        target.num_channels = representation_config.num_channels
        target.pos_encoding_reference_edge = representation_config.pos_encoding_reference_edge
        if hasattr(representation_config, 'material_reconstruction_size'):
            target.material_reconstruction_size = representation_config.material_reconstruction_size
        # Fit the spatial CNN only after normal decoder and layout training.
        decoder_config = copy.deepcopy(target)
        decoder_config.material_cnn_enable = False
        decoder = TCNNModel(decoder_config)
        decoder.astc_proxy = copy.deepcopy(source.astc_proxy)
        from Core.Feature_Grid.compression import freeze_proxies
        with torch.no_grad():
            for old_grid, new_grid in zip(source.feature_grids, decoder.feature_grids):
                new_grid.params.copy_(old_grid.params)
        freeze_proxies(decoder)
        decoder.current_iter = representation["end_iteration"]
        trainer.model, trainer.configs = decoder, decoder_config
        trainer._write_resolved_config(decoder_config)
        print(f"[Multi Stage Training] decoder: {target.n_hidden_layers} hidden layers, width {target.n_neurons}, "
              f"{target.n_frequencies} PE frequencies, {target.feature_gradient_count} feature differences", flush=True)
        start = decoder.current_iter + 1
        details = fit_astc_decoder(trainer, steps=target.multi_stage_training_decoder_steps,
                                   lr_multiplier=target.multi_stage_training_decoder_lr_multiplier,
                                   early_stop=target.early_stop, final_report=False)
        fitted = _record_stage(trainer, stages, "decoder", start, details)

        print("[Multi Stage Training] layout_refit", flush=True)
        details = refit_layout(trainer, target.multi_stage_training_layout_refit_steps)
        if details['applied']:
            start = decoder.current_iter + 1
            decoder.current_iter += details['steps']
            final = _record_stage(trainer, stages, "layout_refit", start, details)
        else:
            stages.append({"name": "layout_refit", "status": "skipped", "details": details})
            _write_stage_report(trainer, stages)
            final = fitted
        if target.material_cnn_enable:
            return _finish_material_cnn(trainer, stages, target.material_cnn_steps)
        trainer.save_compressed_materials(trainer.media_path, final['end_iteration'])
        return _write_stage_report(trainer, stages, completed=True)
    finally:
        trainer.writer.close()


if __name__ == "__main__":
    run_multi_stage_training(get_args())
