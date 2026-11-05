"""Exploratory frozen-checkpoint diagnostics; never alter the training gain."""
import hashlib
import json
from pathlib import Path

import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from EfficientFlow.common.dt_mp1_calibration import calibrate, evaluate, partitions, risks, choose
from EfficientFlow.workspace.train_dt_mp1_workspace import capture, restore, signature


def summarize(scores, budget, alpha):
    # Calibration is already paid for: this measures the lifetime of a fixed gain.
    estimates = risks(scores, budget, overhead=0.)
    result = {}
    for key, rows in scores.items():
        a = np.asarray(rows)
        item = {"gradient_variance":float(a[:, 0].mean()),
                "squared_gradient_drift_unbiased":float(a[:, 1].mean()),
                "gradient_seconds":float(a[:, 2].mean()),
                "budget_feasible":key in estimates}
        if key in estimates:
            item.update(equal_compute_risk=float(estimates[key][0].mean()),
                        batches_per_budget=estimates[key][2])
        if key != "baseline":
            selected, bounds = choose({key:None}, estimates, alpha) if key in estimates else ("baseline", {})
            item.update(diagnostic_fallback_to_baseline=selected == "baseline", upper_bounds=bounds)
        result[key] = item
    return result


def diagnose(checkpoints, output, *, device="cuda:0", pairs=16, workers=0, dataset_path=None):
    """Early gain -> early/middle/late; freshly fitted middle gain -> middle/late.

    Report trajectories are reused across stages for paired diagnostics. They
    must not be used to pick a new deployed gain after inspecting this report.
    """
    if len(checkpoints) != 3 or pairs < 2:
        raise ValueError("Provide early, middle, late checkpoints and at least two pairs")
    paths = [Path(p).expanduser().resolve() for p in checkpoints]
    if len(set(paths)) != 3:
        raise ValueError("Three distinct stage checkpoint files are required")
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device(device)
    saved = capture()
    report = {"statistical_scope":"Exploratory conditional batch-pair diagnostics, not trajectory-population or rollout confidence intervals.",
              "cost_scope":"Full-gradient compute with sunk calibration cost; middle refit cost reported separately. Not an equal GPU-hours training experiment.",
              "training_gain_modified":False, "stages":{}}
    sources = {}
    reference = None
    previous_epoch = -1
    reference_basis = None
    try:
        for stage, path in zip(("early", "middle", "late"), paths):
            payload = torch.load(path, map_location="cpu", pickle_module=dill, weights_only=False)
            if not payload.get("diagnostic_only"):
                raise ValueError("Use dt_early/dt_middle/dt_late diagnostic checkpoints")
            cfg = payload["cfg"]
            current = signature(cfg, common=False)
            epoch = int(payload["epoch"])
            if reference is not None and current != reference:
                raise ValueError("Stage checkpoint configuration mismatch")
            if epoch < previous_epoch:
                raise ValueError("Stage epochs are out of order")
            reference, previous_epoch = current, epoch
            model = hydra.utils.instantiate(cfg.policy)
            model.load_state_dict(payload["state_dicts"]["model"], strict=True)
            model.to(device).eval()
            if reference_basis is not None and not torch.equal(model.dt_U.cpu(), reference_basis):
                raise ValueError("Stage action bases differ; the same K would represent different directions")
            reference_basis = model.dt_U.detach().cpu().clone()
            if stage == "early":
                data_cfg = OmegaConf.create(OmegaConf.to_container(cfg.task.dataset, resolve=True))
                if dataset_path is not None:
                    data_cfg.dataset_path = str(Path(dataset_path).expanduser().resolve())
                ds = hydra.utils.instantiate(data_cfg)
                splits, ids = partitions(ds, cfg.dt.seed)
                report["episodes"] = ids
                bs = int(cfg.dataloader.batch_size)
                batches = min(len(DataLoader(ds, **cfg.dataloader)), cfg.training.max_train_steps or float("inf"))
                if not batches:
                    raise ValueError("No training batches for lifetime cost plan")
                family = payload["dt_selected"]
                gain = model.dt_global if family == "global_scalar" else model.dt_K
                sources["early_gain"] = (family, None if family == "baseline" else gain.detach().cpu().clone())
                report["early_gain"] = {"family":family, "gain":None if family == "baseline" else gain.tolist()}
            if stage == "middle":
                middle_output = output/"middle_refit"
                middle_output.mkdir()
                # A final-epoch midpoint in a tiny smoke run has no deployment
                # horizon; one batch is used only to exercise the diagnostic.
                remaining = max(1, int(batches)*(int(cfg.training.num_epochs)-epoch))
                np.random.seed(cfg.dt.seed); torch.manual_seed(cfg.dt.seed)
                sources["middle_gain"] = calibrate(model, ds, cfg.dt, bs, middle_output, device,
                    workers, remaining_batches=remaining, checkpoint_path=path)
                fam, gain = sources["middle_gain"]
                report["middle_gain"] = {"family":fam, "gain":None if gain is None else gain.tolist(),
                                         "remaining_batches_for_refit":remaining}
            options = {"baseline":("baseline",None), **sources}
            scores, _ = evaluate(model, splits["report"], options, bs, pairs, int(cfg.dt.seed)+400000,
                                 device, "lifetime "+stage, workers)
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda:stream.read(4<<20), b""):
                    digest.update(chunk)
            report["stages"][stage] = {"checkpoint":str(path), "sha256":digest.hexdigest(),
                "epoch":epoch,"global_step":int(payload["global_step"]),"paired_scores":scores,
                "summary":summarize(scores, float(cfg.dt.budget_seconds), float(cfg.dt.alpha))}
            (output/"dt_lifetime.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf8")
            del model, payload
        decisions = [v["diagnostic_fallback_to_baseline"] for stage in report["stages"].values()
                     for k,v in stage["summary"].items() if k != "baseline"]
        report["diagnostic_fallback_fraction"] = float(np.mean(decisions))
        (output/"dt_lifetime.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf8")
        return report
    finally:
        restore(saved)
