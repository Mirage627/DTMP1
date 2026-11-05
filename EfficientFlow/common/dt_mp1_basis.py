"""Fit-only basis discovery for a proxy objective; never a variance guarantee."""
import time
import math
import torch
from EfficientFlow.common.pytorch_util import dict_apply
from EfficientFlow.common.dt_mp1_state import validate_basis


def delta_counts(delta):
    d = delta.detach().cpu()
    return [int((d == 0).sum()), *[int(((d > lo) & (d <= hi)).sum())
            for lo,hi in ((0.,.25),(.25,.5),(.5,.75),(.75,1.))]]


def solve_basis(moment, rank, *, mode="gradient", min_relative_eigenvalue=1e-6,
                min_relative_gap=1e-4):
    """gradient uses sym(E[(H.T ell) v.T]); residual uses E[ell ell.T].

    Absolute eigenvalues rank signed first-order opportunities. This ignores
    curvature; the symmetric score is blind to a purely skew-symmetric moment.
    """
    if moment.ndim != 2 or moment.shape[0] != moment.shape[1] or not 1 <= rank <= len(moment):
        raise ValueError("Invalid DT moment/rank")
    if mode not in ("gradient", "joint_gradient", "residual"):
        raise ValueError("Unknown data-driven DT basis mode")
    if not torch.isfinite(moment).all():
        return None, {"reason":"non-finite basis moment"}
    matrix = (moment.double()+moment.double().T)/2
    extra = {}
    try:
        if mode == "joint_gradient":
            # Weighted union of BOTH singular spaces; never just the left space.
            # Its Gram is M M.T + M.T M. Compressing this union to rank k remains
            # a first-order heuristic, not an optimum of the K regression.
            left, singular, right_h = torch.linalg.svd(moment.double().cpu(),full_matrices=False)
            pool = torch.cat((left*singular, right_h.T*singular),dim=1)
            vectors, energy, _ = torch.linalg.svd(pool,full_matrices=False)
            values = energy.square()
            extra["moment_singular_values"] = singular.tolist()
        else:
            values, vectors = torch.linalg.eigh(matrix.cpu())
    except RuntimeError:
        return None, {"reason":"basis eigensolver failed"}
    order = values.abs().argsort(descending=True)
    values, vectors = values[order], vectors[:,order]
    scale = values.abs()[0].item()
    labels={"gradient":"symmetric first-order proxy","joint_gradient":"joint row/column proxy-score energy",
            "residual":"uncentered residual second moment"}
    info = {"eigenvalues":values.tolist(), "score":labels[mode], **extra}
    if not scale > 0 or values[rank-1].abs() < min_relative_eigenvalue*scale:
        return None, {**info, "reason":"insufficient nonzero spectrum"}
    gap = (values[rank-1].abs()-values[rank].abs()).item()/scale if rank<len(values) else 1.
    info["relative_rank_boundary_gap"] = gap
    if gap < min_relative_gap:
        return None, {**info, "reason":"unstable rank boundary"}
    u = torch.linalg.qr(vectors[:,:rank], mode="reduced")[0]
    # A deterministic sign convention makes saved directions easier to compare.
    signs = u[u.abs().argmax(dim=0), torch.arange(rank)].sign()
    u = (u*signs).float()
    validate_basis(u, len(moment))
    return u, {**info, "reason":"fit candidate; requires independent gradient acceptance"}


def estimate_basis(model, loader, cfg, device):
    from EfficientFlow.policy.dt_mp1 import action_basis
    start = time.perf_counter()
    mode, rank = str(cfg.mode), int(cfg.rank)
    dimension = model.horizon*model.action_dim
    if not 1 <= rank <= dimension or cfg.calibration_split != "fit" or cfg.min_samples < 1:
        raise ValueError("Basis requires a valid rank, fit split, and positive min_samples")
    if not 0 <= float(cfg.min_relative_eigenvalue) <= 1 or not 0 <= float(cfg.min_relative_gap) <= 1:
        raise ValueError("Basis spectral thresholds must be finite fractions in [0,1]")
    info = {"mode":mode, "rank":rank, "active_samples":0, "delta_bins":[0]*5,
            "delta_bin_edges":[0.,.25,.5,.75,1.], "algorithm_version":"dt-basis-v4.1",
            "min_samples":int(cfg.min_samples), "sample_scope":"draws, not independent trajectories"}
    if mode in ("dct_all", "legacy_first4"):
        return action_basis(model.horizon,model.action_dim,mode,rank), {**info,"seconds":time.perf_counter()-start}
    if mode not in ("gradient", "joint_gradient", "residual"):
        raise ValueError(f"Unknown dt.basis.mode {mode}")
    moment = torch.zeros(dimension,dimension,dtype=torch.float64,device=device)
    bucket_moments = torch.zeros(4,dimension,dimension,dtype=torch.float64,device=device)
    for batch in loader:
        rows, delta = model.basis_rows(dict_apply(batch,lambda x:x.to(device)))
        info["delta_bins"] = [a+b for a,b in zip(info["delta_bins"],delta_counts(delta))]
        if rows is None: continue
        v,ell,hte = (a.double() for a in rows)
        moment += hte.T@v if mode in ("gradient","joint_gradient") else ell.T@ell
        active_delta=delta[delta>0]
        for j,(lo,hi) in enumerate(((0.,.25),(.25,.5),(.5,.75),(.75,1.))):
            mask=(active_delta>lo)&(active_delta<=hi)
            bucket_moments[j] += hte[mask].T@v[mask] if mode in ("gradient","joint_gradient") else ell[mask].T@ell[mask]
        info["active_samples"] += len(v)
    n = info["active_samples"]
    def finite_norm(value):
        norm=float(torch.linalg.vector_norm(value))
        return norm if math.isfinite(norm) else None
    info["bucket_mean_score_norms"]=[finite_norm(m/max(1,count)) for m,count in
                                      zip(bucket_moments,info["delta_bins"][1:])]
    info["global_mean_score_norm"]=finite_norm(moment/max(1,n))
    if n < cfg.min_samples:
        u, detail = None, {"reason":"insufficient active basis samples"}
    else:
        u, detail = solve_basis(moment/n,rank,mode=mode,
            min_relative_eigenvalue=float(cfg.min_relative_eigenvalue),
            min_relative_gap=float(cfg.min_relative_gap))
    return u, {**info, **detail, "seconds":time.perf_counter()-start}
