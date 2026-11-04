"""Evaluate an existing MP1 ablation checkpoint and record fresh rollout videos.

Run from the project containing data/, e.g.:
python EfficientFlow/scripts/eval_mp1_checkpoint.py --checkpoint RUN/checkpoints/latest.ckpt \
    --output-dir runs/mp1_reeval --mode offline --n-envs 2
"""
import argparse
import importlib.util
import json
from pathlib import Path
import random
import sys


def evaluate_checkpoint(checkpoint, output_dir, *, device="cuda:0", mode="offline",
                        raw_model=False, n_envs=None, n_train=None, n_test=None,
                        dataset_path=None, num_inference_steps=None, swanlab_module=None):
    import dill
    import hydra
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from EfficientFlow.common.train_entry import register_resolvers
    from EfficientFlow.common.swanlab_logging import (
        SwanLabLogger, scalar_metrics, validate_rollout_videos,
    )

    register_resolvers()
    checkpoint = Path(checkpoint).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if output == checkpoint.parent.parent:
        raise ValueError("Use a separate evaluation output directory to preserve training logs")
    with checkpoint.open("rb") as stream:
        payload = torch.load(stream, map_location="cpu", pickle_module=dill, weights_only=False)
    cfg = OmegaConf.create(OmegaConf.to_container(payload["cfg"], resolve=True))
    state_key = "model" if raw_model or "ema_model" not in payload["state_dicts"] else "ema_model"
    policy = hydra.utils.instantiate(cfg.policy)
    policy.load_state_dict(payload["state_dicts"][state_key], strict=True)
    # Drop raw/EMA duplicates and optimizer moments before creating simulators.
    del payload
    if num_inference_steps is not None:
        from EfficientFlow.common.meanflow_sampling import resolve_time_grid
        resolve_time_grid(num_inference_steps=num_inference_steps)
        policy.num_inference_steps = num_inference_steps
        cfg.policy.num_inference_steps = num_inference_steps
    policy.to(device).eval()
    random.seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)
    torch.manual_seed(cfg.training.seed)
    runner_cfg = cfg.task.env_runner
    for key, value in (("n_envs", n_envs), ("n_train", n_train), ("n_test", n_test)):
        if value is not None:
            if value < (0 if key != "n_envs" else 1):
                raise ValueError(f"Invalid {key}: {value}")
            runner_cfg[key] = value
    for split in ("train", "test"):
        runner_cfg[f"n_{split}_vis"] = min(runner_cfg[f"n_{split}"], runner_cfg[f"n_{split}_vis"])
    if runner_cfg.n_train + runner_cfg.n_test == 0:
        raise ValueError("At least one evaluation episode is required")
    if dataset_path is not None:
        runner_cfg.dataset_path = str(Path(dataset_path).expanduser().resolve())
    cfg.logging.backend, cfg.logging.mode, cfg.logging.id = "swanlab", mode, None
    cfg.logging.name = f"{cfg.get('name', 'MP1')}_reeval_{state_key}_nfe{policy.num_inference_steps or 1}"
    output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output/"eval_config.yaml")
    logger = SwanLabLogger(cfg, output, swanlab_module=swanlab_module)
    runner = None
    try:
        runner = hydra.utils.instantiate(runner_cfg, output_dir=str(output))
        runner.video_backend = "path"
        with torch.no_grad():
            results = runner.run(policy)
        report = scalar_metrics(results)
        expected = runner_cfg.n_train_vis + runner_cfg.n_test_vis
        report["rollout/video_count"] = validate_rollout_videos(results, expected)
        logger.log(report, step=0, rollout=results)
        metadata = {"checkpoint": str(checkpoint), "state_key": state_key, "metrics": report,
                    "configured_nfe": policy.num_inference_steps or 1,
                    "actual_nfe": getattr(policy, "last_sampling_nfe", None),
                    "requested_action_steps": policy.n_action_steps,
                    "effective_action_steps": min(policy.n_action_steps, policy.horizon-policy.n_obs_steps+1)}
        (output/"eval_results.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(json.dumps(metadata, indent=2), flush=True)
        return metadata
    finally:
        try:
            if runner is not None:
                runner.close()
        finally:
            logger.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mode", choices=("online", "offline", "disabled"), default="offline")
    parser.add_argument("--raw-model", action="store_true", help="Use training weights instead of EMA")
    parser.add_argument("--n-envs", type=int)
    parser.add_argument("--n-train", type=int)
    parser.add_argument("--n-test", type=int)
    parser.add_argument("--dataset-path")
    parser.add_argument("--nfe", type=int, nargs="+", help="One count or a sweep, e.g. --nfe 1 2 4 8")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    if "EfficientFlow" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "EfficientFlow", root/"__init__.py", submodule_search_locations=[str(root)])
        package = importlib.util.module_from_spec(spec)
        sys.modules["EfficientFlow"] = package
        spec.loader.exec_module(package)
    counts = args.nfe or [None]
    from EfficientFlow.common.meanflow_sampling import resolve_time_grid
    for count in counts:
        resolve_time_grid(num_inference_steps=count)
    reports = []
    for count in counts:
        output = Path(args.output_dir)/f"nfe_{count}" if len(counts) > 1 else args.output_dir
        reports.append(evaluate_checkpoint(
            args.checkpoint, output, device=args.device, mode=args.mode,
            raw_model=args.raw_model, n_envs=args.n_envs, n_train=args.n_train,
            n_test=args.n_test, dataset_path=args.dataset_path, num_inference_steps=count))
    if len(counts) > 1:
        (Path(args.output_dir)/"nfe_sweep.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
