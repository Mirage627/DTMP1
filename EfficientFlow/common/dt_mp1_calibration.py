"""DT-MP1 v2.1 calibration/selection. All random batches are paired by seed."""
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm
from EfficientFlow.common.pytorch_util import dict_apply
from EfficientFlow.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer


def train_normalizer(ds):
    """Same MP1 per-channel limits transform; held-out episodes never enter stats."""
    out = LinearNormalizer()
    ends = np.asarray(ds.replay_buffer.episode_ends[:])
    starts = np.r_[0, ends[:-1]]
    for key in ["action", *ds.lowdim_keys, *ds.pc_keys]:
        data = ds.replay_buffer[key]
        lo = hi = mean = m2 = None
        count = 0
        for start,end in zip(starts[ds.train_mask], ends[ds.train_mask]):
            for i in range(start,end,128):
                block = np.asarray(data[i:min(i+128,end)],dtype=np.float64).reshape(-1,data.shape[-1])
                if not np.isfinite(block).all():
                    raise ValueError(f"Non-finite training values in {key}")
                n = len(block)
                avg = block.mean(0)
                ss = block.var(0)*n
                if not count:
                    lo,hi,mean,m2 = block.min(0),block.max(0),avg,ss
                else:
                    lo,hi = np.minimum(lo,block.min(0)), np.maximum(hi,block.max(0))
                    diff = avg-mean
                    m2 += ss+diff**2*(count*n/(count+n))
                    mean += diff*(n/(count+n))
                count += n
        if count < 2:
            raise ValueError("Insufficient training data to normalize "+key)
        stats = {k:v.astype(np.float32) for k,v in
                 dict(min=lo,max=hi,mean=mean,std=np.sqrt(m2/(count-1))).items()}
        # Use the original normalizer for scale/offset, including constant channels.
        ref = SingleFieldLinearNormalizer.create_fit(np.stack([stats["min"],stats["max"]]))
        out[key] = SingleFieldLinearNormalizer.create_manual(ref.params_dict["scale"].detach(),
                        ref.params_dict["offset"].detach(),{k:torch.from_numpy(v) for k,v in stats.items()})
    out.requires_grad_(False)
    return out


def partitions(ds, seed):
    held = ds.get_validation_dataset()
    ends = np.asarray(ds.replay_buffer.episode_ends[:])
    episode = np.searchsorted(ends, held.sampler.indices[:,0],side="right")
    ids = np.random.default_rng(seed).permutation(np.unique(episode))
    if len(ids)<4 or ds.train_mask[ids].any():
        raise ValueError("Need at least four held-out episodes disjoint from training")
    groups = dict(zip(("fit","tune","accept","report"),np.array_split(ids,4)))
    subsets = {k:Subset(held,np.flatnonzero(np.isin(episode,v)).tolist()) for k,v in groups.items()}
    return subsets,{k:v.tolist() for k,v in groups.items()}


def families(target):
    order = ["baseline","global_scalar","sub_scalar","diagonal","full"]
    if target not in order:
        raise ValueError(f"Unknown dt.family {target}")
    return order[1:order.index(target)+1]


def fit(policy, loader, names, device):
    k = policy.dt_K.shape[0]
    gram, rhs, count = {}, {}, 0
    for batch in tqdm(loader,desc="DT fit: frozen spatial JVPs",leave=False):
        rows = policy.calibration_rows(dict_apply(batch,lambda x:x.to(device)))
        if rows is None:
            continue
        R,p,hv,ell = (x.double() for x in rows)
        count += len(R)
        diag = R*p[:,None,:]
        for name in names:
            if name=="diagonal": X = diag
            elif name=="sub_scalar": X = diag.sum(-1,keepdim=True)/math.sqrt(k)
            elif name=="global_scalar": X = hv[:,:,None]
            else: X = (R[:,:,:,None]*p[:,None,None,:]).flatten(2)
            design = X.reshape(-1,X.shape[-1])
            gram[name] = gram.get(name,0) + design.T@design
            rhs[name] = rhs.get(name,0) + design.T@ell.flatten()
    options = {"baseline":("baseline",None)}
    for name in names:
        if not count: continue
        C, f = gram[name].cpu()/count,rhs[name].cpu()/count
        scale = C.diag().mean()
        if not (torch.isfinite(C).all() and torch.isfinite(f).all() and scale>0): continue
        for rho in (.01,.1,1.,10.):
            try:
                v = torch.linalg.solve(C+rho*scale*torch.eye(len(f),dtype=f.dtype),f)
            except RuntimeError:
                continue
            if not torch.isfinite(v).all(): continue
            if name=="global_scalar": gain=v[0]
            elif name=="sub_scalar": gain=torch.eye(k,dtype=v.dtype)*v[0]/math.sqrt(k)
            elif name=="diagonal": gain=torch.diag(v)
            else: gain=v.reshape(k,k)
            size = gain.abs() if gain.ndim==0 else torch.linalg.matrix_norm(gain,ord=2)
            if size<=2 and size>0: options[f"{name}:{rho:g}"]=(name,gain.float())
    return options,count


def paired_batches(ds, count, bs, seed, workers=0):
    # Independent minibatches, but like the shuffled MP1 loader each batch is
    # drawn WITHOUT replacement internally (dispersive loss couples its rows).
    if len(ds)<bs:
        raise ValueError(f"DT split has only {len(ds)} windows for batch size {bs}; "
                         "increase held-out trajectories or use a smaller common batch size")
    rng=np.random.default_rng(seed)
    indices=[rng.choice(len(ds),size=bs,replace=False).tolist() for _ in range(count)]
    return DataLoader(ds,batch_sampler=indices,num_workers=workers,
                      generator=torch.Generator().manual_seed(seed))


def gradients(model, batch, seed, device, diagnostic="normal"):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model.zero_grad(set_to_none=True)
    if device.type=="cuda": torch.cuda.synchronize(device)
    start=time.perf_counter()
    loss,_=model.compute_loss(batch,diagnostic=diagnostic)
    loss.backward()
    if device.type=="cuda": torch.cuda.synchronize(device)
    elapsed=time.perf_counter()-start
    # Detach onto host; keep only two reference gradients and one candidate pair.
    grad=[v.grad.detach().to(device="cpu",copy=True) for v in model.parameters() if v.grad is not None]
    model.zero_grad(set_to_none=True)
    return grad,elapsed


def pair_stats(left,right,base_l,base_r):
    if not len(left)==len(right)==len(base_l)==len(base_r):
        raise RuntimeError("Parameter gradient layout changed between DT candidates")
    var=drift=0.
    for L,R,B,C in zip(left,right,base_l,base_r):
        # Chunk large convolutions; no dense model-sized covariance or FP64 copy.
        for l,r,b,c in zip(*(v.reshape(-1).split(262144) for v in (L,R,B,C))):
            d=(l-r).double()
            var+=.5*torch.dot(d,d).item()
            drift+=torch.dot((l-b).double(),(r-c).double()).item()
    if not math.isfinite(var+drift): raise FloatingPointError("Non-finite DT gradient statistic")
    return var,drift


def evaluate(model,ds,options,bs,pairs,seed,device,stage,workers=0,mechanism=False):
    if pairs<2: raise ValueError("Need at least two independent pairs")
    data=iter(paired_batches(ds,2*pairs,bs,seed,workers))
    scores={k:[] for k in options}
    mech={k:[] for k in options} if mechanism else {}
    for i in tqdm(range(pairs),desc=f"DT {stage}: paired full gradients"):
        batches=[dict_apply(next(data),lambda x:x.to(device)) for _ in range(2)]
        def measure(diag="normal"):
            return [gradients(model,batch,seed+2*i+j+1,device,diag)
                    for j,batch in enumerate(batches)]
        model.set_gain("baseline")
        (bl,tl),(br,tr)=measure()
        if mechanism:
            (gl,_),(gr,_)=measure("time")
            time_var=pair_stats(gl,gr,gl,gr)[0]
            (nl,_),(nr,_)=measure("none")
            none_var=pair_stats(nl,nr,nl,nr)[0]
            temporal_var=0.
            for L,R,P,Q in zip(gl,gr,nl,nr):
                for a,b,c,d in zip(*(v.reshape(-1).split(262144) for v in (L,R,P,Q))):
                    v=((a-c)-(b-d)).double()
                    temporal_var+=.5*torch.dot(v,v).item()
            del nl,nr
        for key,(fam,k) in options.items():
            model.set_gain(fam,k)
            if key=="baseline": l,r,c1,c2=bl,br,tl,tr
            else: (l,c1),(r,c2)=measure()
            var,drift=pair_stats(l,r,bl,br)
            scores[key].append([var,drift,(c1+c2)/2])
            if mechanism:
                diff_var=0.
                for L,R,P,Q in zip(l,r,gl,gr):
                    for a,b,c,d in zip(*(v.reshape(-1).split(262144) for v in (L,R,P,Q))):
                        v=((a-c)-(b-d)).double()
                        diff_var+=.5*torch.dot(v,v).item()
                mech[key].append({"time":time_var,"spatial_difference":diff_var,
                                  "spatial_cross":var-time_var-diff_var,"none":none_var,
                                  "temporal_difference":temporal_var,
                                  "temporal_cross":time_var-none_var-temporal_var})
            del l,r
    model.set_gain("baseline")
    return scores,mech


def risks(scores, budget, overhead, locked_costs=None):
    out={}
    for key,rows in scores.items():
        a=np.asarray(rows)
        avail=budget-(0 if key=="baseline" else overhead)
        cost=a[:,2].mean() if locked_costs is None else locked_costs[key]
        n=math.floor(avail/cost)
        if n>=1: out[key]=(a[:,0]/n+a[:,1],a[:,1],n)
    if "baseline" not in out: raise ValueError("dt.budget_seconds is shorter than one baseline batch")
    return out


def choose(locked, risks_, alpha):
    # Paired, one-sided approximate basic bootstrap on independent *batch pairs*,
    # conditional on this empirical dataset and measured timings, not a rollout CI.
    base=risks_["baseline"][0]
    selected="baseline"
    checks={}
    scalar_keys=[k for k in locked if k in risks_ and k.startswith(("global_scalar:","sub_scalar:"))]
    strong=locked.get("strong",None)
    keys=[k for k in locked if k!="strong"]
    comparisons=sum(2+len(scalar_keys)+(1 if k.startswith("full:") else 0) for k in keys)
    indices=np.random.default_rng(748323).integers(len(base),size=(16384,len(base)))
    def upper(v):
        return float(2*v.mean()-np.quantile(v[indices].mean(1),alpha/max(1,comparisons)))
    for key in keys:
        if key not in risks_: continue
        risk,drift,_=risks_[key]
        against=["baseline"]
        if not key.startswith(("global_scalar:","sub_scalar:")):
            against+=scalar_keys
            if key.startswith("full:"):
                against += [k for k in keys if k.startswith("diagonal:") and k in risks_]
        elif key!=strong: continue
        checks[key]={k:upper(risk-risks_[k][0]) for k in against}
        checks[key]["drift_minus_tolerance"]=upper(drift-base)
        if all(checks[key][k]<0 for k in against) and checks[key]["drift_minus_tolerance"]<=0:
            selected=key
    return selected,checks


def calibrate(model,ds,cfg,bs,output,device,workers=0):
    names=families(cfg.family)
    if not names: return "baseline",None
    start=time.perf_counter()
    path=Path(output)/"dt_calibration.json"
    def save():
        tmp=path.with_suffix(".tmp")
        tmp.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf8")
        os.replace(tmp,path)
    if cfg.amortization_windows<1 or not 0<cfg.alpha<1 or cfg.fit_batches<1:
        raise ValueError("Invalid DT calibration settings")
    splits,ids=partitions(ds,int(cfg.seed))
    seed=int(cfg.seed)
    if device.type=="cuda": torch.cuda.reset_peak_memory_stats(device)
    options,n=fit(model,paired_batches(splits["fit"],cfg.fit_batches,bs,seed,workers),names,device)
    digest=hashlib.sha256()
    with (Path(output)/"checkpoints/dt_warmup.ckpt").open("rb") as fp:
        for chunk in iter(lambda:fp.read(4<<20),b""): digest.update(chunk)
    channels=(["x","y","z"]+[f"rotation_6d[{i}]" for i in range(6)]+["gripper"]
              if model.action_dim==10 else [f"action[{i}]" for i in range(model.action_dim)])
    report={"episodes":{"train":np.flatnonzero(ds.train_mask).tolist(),**ids},"seed":seed,
            "n_active_fit":n,"w":model.w,"basis":model.dt_U.cpu().tolist(),
            "action_order":"normalized sequence, time-major; channels in dataset rotation_6d order",
            "candidates":{k:None if v is None else v.tolist() for k,(_,v) in options.items()},
            "checkpoint":"checkpoints/dt_warmup.ckpt","checkpoint_sha256":digest.hexdigest(),
            "action_channels":channels,"amortization_windows":cfg.amortization_windows,
            "budget_seconds":cfg.budget_seconds,"alpha":cfg.alpha,
            "code_sha256":{str(p.name):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                [Path(__file__),Path(__file__).parents[1]/"policy/dt_mp1.py"]}}
    save()
    if len(options)==1:
        report.update(selected="baseline",reason="no valid nonzero fit"); save()
        return "baseline",None
    tune,_=evaluate(model,splits["tune"],options,bs,cfg.tune_pairs,seed+100000,device,"tune",workers)
    cost=lambda: (time.perf_counter()-start)/cfg.amortization_windows
    tr=risks(tune,cfg.budget_seconds,cost())
    winners={}
    for fam in names:
        # Zero is admissible in every family.
        keys=[k for k in tr if k.startswith(fam+":")]+["baseline"]
        best=min(keys,key=lambda k:tr[k][0].mean())
        if best!="baseline": winners[best]=options[best]
    scalars=[k for k in winners if k.startswith(("global_scalar:","sub_scalar:"))]
    strong=min(scalars,key=lambda k:tr[k][0].mean()) if scalars else None
    fixed={"baseline":options["baseline"],**winners}
    costs={k:float(np.mean(np.asarray(tune[k])[:,2])) for k in fixed}
    report.update(tune=tune,locked=list(fixed),strong_scalar=strong,locked_gradient_seconds=costs); save()
    acceptance,_=evaluate(model,splits["accept"],fixed,bs,cfg.accept_pairs,seed+200000,device,"accept",workers)
    allocated=cost()
    ar=risks(acceptance,cfg.budget_seconds,allocated,costs)
    # If the requested comparator had insufficient budget its superiority cannot be established.
    elig={k:None for k in winners if k in ar}
    if any(k not in ar for k in scalars):
        elig={k:None for k in elig if k in scalars}
    if any(k.startswith("diagonal:") and k not in ar for k in winners):
        elig={k:None for k in elig if not k.startswith("full:")}
    selected,checks=choose({**elig,"strong":strong},ar,cfg.alpha)
    report.update(acceptance=acceptance,upper_bounds=checks,selected=selected,
                  seconds_through_decision=time.perf_counter()-start,allocated_seconds=allocated,
                  independent_batches_per_budget={k:v[2] for k,v in ar.items()}); save()
    test,mech=evaluate(model,splits["report"],fixed,bs,cfg.report_pairs,seed+300000,device,"report",workers,True)
    report.update(locked_test=test,spatial_decomposition_time_difference_cross=mech,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None,
                  total_seconds=time.perf_counter()-start); save()
    print(f"DT accepted: {selected}; audit: {path}",flush=True)
    return options[selected]
