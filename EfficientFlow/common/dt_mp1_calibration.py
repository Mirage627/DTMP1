"""DT-MP1 calibration/selection. Conditional batch-pair diagnostics, not rollout CIs."""
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
from EfficientFlow.common.dt_mp1_state import Correction, normalizer_fingerprint
from EfficientFlow.common.dt_mp1_basis import estimate_basis, delta_counts


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


def fit(policy, loader, names, device, gain_norm_limit=2.):
    k = policy.dt_K.shape[0]
    gram, rhs, count = {}, {}, 0
    for batch in tqdm(loader,desc="DT fit: frozen spatial JVPs",leave=False):
        batch=dict_apply(batch,lambda x:x.to(device))
        rows = (policy.calibration_rows(batch,need_basis=False) if names==["global_scalar"] else
                policy.calibration_rows(batch))
        if rows is None:
            continue
        R,p,hv,ell = (x.double() for x in rows)
        count += len(R)
        diag = R*p[:,None,:] if R.shape[-1] else None
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
            if size<=gain_norm_limit and size>0: options[f"{name}:{rho:g}"]=(name,gain.float())
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


def evaluate(model,ds,options,bs,pairs,seed,device,stage,workers=0,mechanism=False,max_grad_norm=None):
    original = Correction.capture(model)
    options = {key:(value if isinstance(value,Correction) else
                   Correction(original.basis, *value, original.metadata)) for key,value in options.items()}
    try:
        return _evaluate(model,ds,options,bs,pairs,seed,device,stage,workers,mechanism,max_grad_norm)
    finally:
        model.set_correction(original)


def gradient_quality(left,right,base_l,base_r,max_grad_norm):
    result=[]
    for candidate,base in ((left,base_l),(right,base_r)):
        norm2=base2=dot=0.
        for a,b in zip(candidate,base):
            for x,y in zip(a.reshape(-1).split(262144),b.reshape(-1).split(262144)):
                x,y=x.double(),y.double()
                norm2+=torch.dot(x,x).item(); base2+=torch.dot(y,y).item(); dot+=torch.dot(x,y).item()
        norm=math.sqrt(norm2)
        result.append({"gradient_norm":norm,"cosine_to_baseline":dot/math.sqrt(norm2*base2) if norm2*base2 else None,
                       "would_clip":None if max_grad_norm is None else norm>max_grad_norm,
                       "clip_scale":1. if max_grad_norm is None else min(1.,max_grad_norm/(norm+1e-6))})
    return result


def _evaluate(model,ds,options,bs,pairs,seed,device,stage,workers,mechanism,max_grad_norm):
    if pairs<2: raise ValueError("Need at least two independent pairs")
    data=iter(paired_batches(ds,2*pairs,bs,seed,workers))
    scores={k:[] for k in options}
    mech={k:[] for k in options}
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
        for key,correction in options.items():
            fam=correction.family
            model.set_correction(correction)
            if key=="baseline" or fam=="baseline": l,r,c1,c2=bl,br,tl,tr
            else: (l,c1),(r,c2)=measure()
            var,drift=pair_stats(l,r,bl,br)
            scores[key].append([var,drift,(c1+c2)/2])
            detail={"gradients":gradient_quality(l,r,bl,br,max_grad_norm)}
            if mechanism:
                diff_var=0.
                for L,R,P,Q in zip(l,r,gl,gr):
                    for a,b,c,d in zip(*(v.reshape(-1).split(262144) for v in (L,R,P,Q))):
                        v=((a-c)-(b-d)).double()
                        diff_var+=.5*torch.dot(v,v).item()
                detail.update({"time":time_var,"spatial_difference":diff_var,
                                  "spatial_cross":var-time_var-diff_var,"none":none_var,
                                  "temporal_difference":temporal_var,
                                  "temporal_cross":time_var-none_var-temporal_var})
            mech[key].append(detail)
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


def allocated_overhead(seconds, remaining_batches, baseline_seconds, budget):
    """Allocate calibration cost to a baseline-time window of the remaining run."""
    if remaining_batches < 1 or baseline_seconds <= 0 or budget <= 0 or seconds < 0:
        raise ValueError("Invalid remaining-training cost plan")
    values = (seconds, remaining_batches, baseline_seconds, budget)
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Non-finite cost plan")
    return seconds * min(1., budget / (remaining_batches * baseline_seconds))


def choose(locked, risks_, alpha):
    # Paired, one-sided approximate basic bootstrap on independent *batch pairs*,
    # conditional on this empirical dataset and measured timings, not a rollout CI.
    base=risks_["baseline"][0]
    selected="baseline"
    checks={}
    def family(key):
        value=locked[key]
        return value.family if isinstance(value,Correction) else value
    scalar_keys=[k for k in locked if k in risks_ and family(k) in ("global_scalar","sub_scalar")]
    strong=locked.get("strong",None)
    keys=[k for k in locked if k!="strong"]
    comparisons=sum(2+len(scalar_keys)+(1 if family(k)=="full" else 0) for k in keys)
    indices=np.random.default_rng(748323).integers(len(base),size=(16384,len(base)))
    def upper(v):
        return float(2*v.mean()-np.quantile(v[indices].mean(1),alpha/max(1,comparisons)))
    for key in keys:
        if key not in risks_: continue
        risk,drift,_=risks_[key]
        against=["baseline"]
        if family(key) not in ("global_scalar","sub_scalar"):
            against+=scalar_keys
            if family(key)=="full":
                against += [k for k in keys if family(k)=="diagonal" and k in risks_]
        elif key!=strong: continue
        checks[key]={k:upper(risk-risks_[k][0]) for k in against}
        checks[key]["drift_minus_tolerance"]=upper(drift-base)
        if all(checks[key][k]<0 for k in against) and checks[key]["drift_minus_tolerance"]<=0:
            selected=key
    return selected,checks


def calibrate(model,ds,cfg,bs,output,device,workers=0,*,remaining_batches=None,checkpoint_path=None,max_grad_norm=None):
    original=Correction.capture(model)
    training=model.training
    try:
        model.eval()
        model.set_gain("baseline")
        return _calibrate(model,ds,cfg,bs,output,device,workers,remaining_batches,checkpoint_path,max_grad_norm)
    finally:
        model.set_correction(original)
        model.train(training)


def relative_safety(model,ds,options,cfg,bs,seed,device,workers):
    """Pre-registered fit-only tail rejection, never tuned on accept/report data."""
    retained={"baseline":options["baseline"]}
    report={}
    for key,correction in options.items():
        if key=="baseline": continue
        model.set_correction(correction)
        ratios=[]; counts=[0]*5
        for i,batch in enumerate(paired_batches(ds,cfg.fit_batches,bs,seed,workers)):
            random.seed(seed+i); np.random.seed(seed+i); torch.manual_seed(seed+i)
            ratio,delta=model.correction_ratios(dict_apply(batch,lambda x:x.to(device)))
            ratios.append(ratio.cpu())
            counts=[a+b for a,b in zip(counts,delta_counts(delta))]
        ratio=torch.cat(ratios)
        valid=len(ratio)>=int(cfg.basis.min_samples) and bool(torch.isfinite(ratio).all())
        q=float(torch.quantile(ratio.double(),float(cfg.relative_correction_quantile))) if valid else None
        accepted=valid and q<=float(cfg.max_relative_correction)
        report[key]={"quantile":q,"accepted":accepted,"active_samples":len(ratio),"delta_bins":counts,
                     "reason":"within fit safety limit" if accepted else "insufficient samples or excessive/nonfinite correction"}
        if accepted: retained[key]=correction
    model.set_gain("baseline")
    return retained,report


def fit_basis_candidates(model,ds,cfg,bs,seed,device,workers,bases,names,metadata):
    """Refit K for each frozen U; same fit draws; one independent global scalar."""
    original=Correction.capture(model)
    options={"baseline":Correction(original.basis,"baseline",metadata=metadata)}
    counts={}
    plans={"global":(original.basis,{"mode":"unused_global_scalar"},["global_scalar"])}
    plans.update({key:(u,info,[n for n in names if n!="global_scalar"])
                  for key,(u,info) in bases.items() if u is not None})
    try:
        for label,(u,info,fit_names) in plans.items():
            if not fit_names: continue
            model.set_gain("baseline"); model.set_basis(u)
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            fitted,n=fit(model,paired_batches(ds,cfg.fit_batches,bs,seed,workers),fit_names,device,cfg.gain_norm_limit)
            counts[label]=n
            meta={**metadata,"basis_fit":info,"rank":model.dt_U.shape[1],"basis_candidate":label}
            for key,(fam,gain) in fitted.items():
                if key=="baseline": continue
                name=key if label=="global" else label+"/"+key
                options[name]=Correction(model.dt_U.detach().cpu().clone(),fam,gain,meta)
    finally:
        model.set_correction(original)
    return options,counts


def _calibrate(model,ds,cfg,bs,output,device,workers,remaining_batches,checkpoint_path,max_grad_norm):
    names=families(cfg.family)
    if not names: return Correction.capture(model,{"reason":"configured baseline"})
    start=time.perf_counter()
    path=Path(output)/"dt_calibration.json"
    def save():
        tmp=path.with_suffix(".tmp")
        tmp.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf8")
        os.replace(tmp,path)
    if remaining_batches is None or remaining_batches < 1 or not 0<cfg.alpha<1 or cfg.fit_batches<1:
        raise ValueError("Invalid DT calibration settings")
    if (not 0<float(cfg.gain_norm_limit)<=2 or not 0<float(cfg.relative_correction_quantile)<1
            or not 0<float(cfg.max_relative_correction)<float("inf")):
        raise ValueError("Invalid pre-registered DT gain/correction limits")
    splits,ids=partitions(ds,int(cfg.seed))
    seed=int(cfg.seed)
    if device.type=="cuda": torch.cuda.reset_peak_memory_stats(device)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    bases={}
    if names!=["global_scalar"]:
        u,basis_info=estimate_basis(model,paired_batches(splits["fit"],cfg.fit_batches,bs,seed,workers),cfg.basis,device)
        bases[f"{cfg.basis.mode}:r{cfg.basis.rank}"]=(u,basis_info)
        if cfg.basis.compare_dct and not (cfg.basis.mode=="dct_all" and cfg.basis.rank==model.action_dim):
            from EfficientFlow.policy.dt_mp1 import action_basis
            static=action_basis(model.horizon,model.action_dim,"dct_all",model.action_dim)
            bases[f"dct_all:r{model.action_dim}"]=(static,{"mode":"dct_all","rank":model.action_dim,
                "reason":"pre-registered whole-channel constant-time comparator","seconds":0.})
    digest=hashlib.sha256()
    checkpoint_path=Path(checkpoint_path or Path(output)/"checkpoints/dt_warmup.ckpt")
    with checkpoint_path.open("rb") as fp:
        for chunk in iter(lambda:fp.read(4<<20),b""): digest.update(chunk)
    metadata={"normalizer_sha256":normalizer_fingerprint(model),"checkpoint_sha256":digest.hexdigest(),
              "seed":seed,
              "action_order":"normalized time-major; dataset rotation_6d channel order"}
    fit_start=time.perf_counter()
    # R is collected only AFTER each U is fixed. No old-basis cache is reused.
    options,counts=fit_basis_candidates(model,splits["fit"],cfg,bs,seed+1,device,workers,bases,names,metadata)
    fit_seconds=time.perf_counter()-fit_start
    fitted_candidates={k:v.as_dict() for k,v in options.items()}
    safety_start=time.perf_counter()
    options,safety=relative_safety(model,splits["fit"],options,cfg,bs,seed+2,device,workers)
    safety_seconds=time.perf_counter()-safety_start
    channels=(["x","y","z"]+[f"rotation_6d[{i}]" for i in range(6)]+["gripper"]
              if model.action_dim==10 else [f"action[{i}]" for i in range(model.action_dim)])
    report={"episodes":{"train":np.flatnonzero(ds.train_mask).tolist(),**ids},"seed":seed,
            "n_active_fit_by_basis":counts,"w":model.w,"basis":None,
            "action_order":"normalized sequence, time-major; channels in dataset rotation_6d order",
            "candidates":{k:v.as_dict() for k,v in options.items()},
            "fitted_candidates":fitted_candidates,
            "basis_fit":{k:info for k,(_,info) in bases.items()},"normalizer_sha256":metadata["normalizer_sha256"],
            "relative_correction_safety":safety,"fit_seconds":fit_seconds,"safety_seconds":safety_seconds,
            "max_relative_correction":float(cfg.max_relative_correction),
            "relative_correction_quantile":float(cfg.relative_correction_quantile),
            "checkpoint":str(checkpoint_path.resolve()),"checkpoint_sha256":digest.hexdigest(),
            "action_channels":channels,"remaining_training_batches":int(remaining_batches),
            "budget_seconds":cfg.budget_seconds,"alpha":cfg.alpha,
            "statistical_scope":"Approximate batch-pair bootstrap conditional on fixed episodes and measured timings; not a population or rollout guarantee.",
            "cost_scope":"Measured full-gradient compute only; excludes training data loading, optimizer, EMA and rollout. Full calibration/report time is charged; not a measured GPU-hours training comparison.",
            "code_sha256":{str(p.name):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                [Path(__file__),Path(__file__).with_name("dt_mp1_basis.py"),
                 Path(__file__).with_name("dt_mp1_state.py"),Path(__file__).parents[1]/"policy/dt_mp1.py"]}}
    save()
    if len(options)==1:
        report.update(selected="baseline",reason="no nonzero candidate passed fit/safety checks",
                      basis=options["baseline"].basis.tolist(),
                      total_seconds=time.perf_counter()-start); save()
        return options["baseline"]
    tune,tune_quality=evaluate(model,splits["tune"],options,bs,cfg.tune_pairs,seed+100000,device,"tune",workers,
                              max_grad_norm=max_grad_norm)
    baseline_seconds=float(np.mean(np.asarray(tune["baseline"])[:,2]))
    cost=lambda: allocated_overhead(time.perf_counter()-start,remaining_batches,
                                    baseline_seconds,cfg.budget_seconds)
    tr=risks(tune,cfg.budget_seconds,cost())
    winners={}
    for fam in names:
        # Zero is admissible in every family.
        keys=[k for k in tr if options[k].family==fam]+["baseline"]
        best=min(keys,key=lambda k:tr[k][0].mean())
        if best!="baseline": winners[best]=options[best]
    scalars=[k for k in winners if winners[k].family in ("global_scalar","sub_scalar")]
    strong=min(scalars,key=lambda k:tr[k][0].mean()) if scalars else None
    fixed={"baseline":options["baseline"],**winners}
    costs={k:float(np.mean(np.asarray(tune[k])[:,2])) for k in fixed}
    report.update(tune=tune,tune_gradient_quality=tune_quality,locked=list(fixed),strong_scalar=strong,locked_gradient_seconds=costs); save()
    # Charge actual report wall time. Its gradient statistics NEVER enter the gate.
    test,mech=evaluate(model,splits["report"],fixed,bs,cfg.report_pairs,seed+300000,device,"report",workers,True,max_grad_norm)
    acceptance,accept_quality=evaluate(model,splits["accept"],fixed,bs,cfg.accept_pairs,seed+200000,device,"accept",workers,
                                       max_grad_norm=max_grad_norm)
    allocated=cost()
    ar=risks(acceptance,cfg.budget_seconds,allocated,costs)
    # If the requested comparator had insufficient budget its superiority cannot be established.
    elig={k:winners[k] for k in winners if k in ar}
    if any(k not in ar for k in scalars):
        elig={k:v for k,v in elig.items() if k in scalars}
    if any(winners[k].family=="diagonal" and k not in ar for k in winners):
        elig={k:v for k,v in elig.items() if v.family!="full"}
    selected,checks=choose({**elig,"strong":strong},ar,cfg.alpha)
    report.update(acceptance=acceptance,accept_gradient_quality=accept_quality,upper_bounds=checks,selected=selected,
                  basis=options[selected].basis.tolist(),
                  seconds_through_decision=time.perf_counter()-start,allocated_seconds=allocated,
                  independent_batches_per_budget={k:v[2] for k,v in ar.items()}); save()
    report.update(locked_test=test,spatial_decomposition_time_difference_cross=mech,
                  baseline_gradient_seconds=baseline_seconds,
                  estimated_remaining_baseline_gradient_seconds=remaining_batches*baseline_seconds,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None,
                  total_seconds=time.perf_counter()-start); save()
    print(f"DT accepted: {selected}; audit: {path}",flush=True)
    return options[selected]
