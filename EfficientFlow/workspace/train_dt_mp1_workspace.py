"""Isolated MP1 trainer: common warm-up -> locked DT fit/gate -> continuation."""
import copy
import json
import math
import os
import random
from pathlib import Path
import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from EfficientFlow.common.dt_mp1_calibration import calibrate, families, partitions, train_normalizer
from EfficientFlow.common.pytorch_util import dict_apply, optimizer_to
from EfficientFlow.common.swanlab_logging import SwanLabLogger, scalar_metrics
from EfficientFlow.model.common.lr_scheduler import get_scheduler
from EfficientFlow.workspace.base_workspace import BaseWorkspace


def capture():
    return (random.getstate(),np.random.get_state(),torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore(rng):
    random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2])
    if rng[3] is not None: torch.cuda.set_rng_state_all(rng[3])


def preflight_hdf5(cfg):
    """Check metadata before encoding hundreds of demonstrations into the replay buffer."""
    import h5py
    path=Path(cfg.task.dataset.dataset_path).expanduser()
    if not path.is_file(): raise FileNotFoundError(f"DT-MP1 dataset not found: {path.resolve()}")
    meta=cfg.task.dataset.shape_meta
    if not cfg.task.dataset.abs_action or cfg.task.dataset.rotation_rep!="rotation_6d":
        raise ValueError("Expected MP1 absolute actions with rotation_6d")
    with h5py.File(path,"r") as f:
        demos=f["data"]
        for i in range(int(cfg.n_demo)):
            key=f"demo_{i}"
            if key not in demos: raise ValueError(f"{path}: requested {cfg.n_demo} demos; missing {key}")
            demo=demos[key]
            shape=demo["actions"].shape
            if len(shape)!=2 or not shape[0] or (shape[1],meta.action["shape"][0]) not in ((7,10),(14,20)):
                raise ValueError(f"{key}: raw action shape {shape} cannot encode to {meta.action['shape']}")
            for field,spec in meta.obs.items():
                dims=tuple(spec["shape"])
                if spec.get("type")=="rgb": dims=(dims[1],dims[2],dims[0])
                actual=demo["obs"][field].shape
                if actual!=(shape[0],*dims):
                    raise ValueError(f"{key}/obs/{field}: {actual}; expected {(shape[0],*dims)}")
    print(f"DT data OK: {path.resolve()}, {cfg.n_demo} demonstrations; six-channel point clouds",flush=True)


def signature(cfg, common):
    plain=OmegaConf.to_container(cfg,resolve=True)
    vals={k:plain[k] for k in ("policy","task_name","n_demo","task","dataloader","optimizer","ema")}
    vals["training"]={k:plain["training"].get(k) for k in (
        "seed","num_epochs","max_train_steps","lr_scheduler","lr_warmup_steps","gradient_accumulate_every",
        "max_grad_norm","use_ema","rollout_every","enable_rollouts")}
    vals["dt"]={k:v for k,v in plain["dt"].items() if k!="warmup_checkpoint" and
                (not common or k=="warmup_epochs")}
    return vals


class TrainDTMP1Workspace(BaseWorkspace):
    include_keys=("epoch","global_step","dt_selected","rng","best_score")
    def __init__(self,cfg,output_dir=None):
        super().__init__(cfg,output_dir)
        random.seed(cfg.training.seed); np.random.seed(cfg.training.seed); torch.manual_seed(cfg.training.seed)
        self.model=hydra.utils.instantiate(cfg.policy)
        self.ema_model=copy.deepcopy(self.model) if cfg.training.use_ema else None
        self.optimizer=hydra.utils.instantiate(cfg.optimizer,params=self.model.parameters())
        self.epoch=self.global_step=0
        self.dt_selected=None
        self.best_score=-math.inf
        self.rng=None
        self.scheduler=None

    def _save(self,tag):
        path=self.get_checkpoint_path(tag)
        tmp=path.with_suffix(".partial")
        self.rng=capture()
        path.parent.mkdir(parents=True,exist_ok=True)
        # Same payload layout as BaseWorkspace, synchronously close before rename.
        payload={"cfg":self.cfg,
                 "state_dicts":{k:v.state_dict() for k,v in self.__dict__.items()
                                if hasattr(v,"state_dict") and hasattr(v,"load_state_dict")},
                 "pickles":{k:dill.dumps(getattr(self,k)) for k in (*self.include_keys,"_output_dir")}}
        with tmp.open("wb") as fp: torch.save(payload,fp,pickle_module=dill)
        os.replace(tmp,path)

    def load_payload(self,payload,exclude_keys=None,include_keys=None,**kwargs):
        # Never redirect a fork to the warmup run's output directory.
        keys=list(self.include_keys) if include_keys is None else include_keys
        skip=list(exclude_keys or ())
        if self.scheduler is None: skip.append("scheduler")
        super().load_payload(payload,exclude_keys=skip,include_keys=keys,**kwargs)

    def run(self):
        cfg=self.cfg
        train=cfg.training
        families(cfg.dt.family)
        if not 1<=cfg.dt.warmup_epochs<train.num_epochs:
            raise ValueError("Need at least one warm-up epoch and one post-calibration epoch")
        if min(train.rollout_every,train.checkpoint_every,train.gradient_accumulate_every)<1:
            raise ValueError("Invalid training interval")
        if cfg.n_latency_steps or cfg.n_action_steps+cfg.n_obs_steps-1>cfg.horizon:
            raise ValueError("DT-MP1 requires a valid zero-latency MP1 action window")
        output=Path(self.output_dir)
        output.mkdir(parents=True,exist_ok=True)
        device=torch.device(train.device)
        if cfg.task.dataset._target_.endswith("RobomimicReplayPointCloudDataset"):
            preflight_hdf5(cfg)
        ds=hydra.utils.instantiate(cfg.task.dataset)
        loader=DataLoader(ds,**cfg.dataloader)
        batches=min(len(loader),train.max_train_steps or len(loader))
        if not batches or cfg.dataloader.batch_size<2: raise ValueError("MP1 needs a nonempty batch of at least 2")
        if cfg.dt.family!="baseline":
            splits,_=partitions(ds,cfg.dt.seed)
            if any(len(s)<cfg.dataloader.batch_size for s in splits.values()):
                raise ValueError("Each held-out DT partition needs at least a full, distinct-window minibatch")
        accum=int(train.gradient_accumulate_every)
        # Bind AdamW to the final-device Parameters, including on PyTorch
        # versions that replace Parameter objects during a CPU -> CUDA move.
        self.model.to(device)
        self.optimizer=hydra.utils.instantiate(cfg.optimizer,params=self.model.parameters())
        self.scheduler=get_scheduler(train.lr_scheduler,optimizer=self.optimizer,
                num_warmup_steps=train.lr_warmup_steps,num_training_steps=math.ceil(batches/accum)*train.num_epochs)
        latest=self.get_checkpoint_path()
        resume=train.resume and latest.is_file()
        if latest.is_file() and not resume:
            raise FileExistsError(f"Use a fresh run directory or resume {latest}")
        if resume or cfg.dt.warmup_checkpoint:
            payload=torch.load(latest if resume else cfg.dt.warmup_checkpoint,map_location="cpu",pickle_module=dill)
            if signature(cfg,not resume)!=signature(payload["cfg"],not resume):
                raise ValueError("Checkpoint dataset, policy or training configuration mismatch")
            self.load_payload(payload)
            del payload
            if not resume and (self.dt_selected is not None or self.epoch!=cfg.dt.warmup_epochs):
                raise ValueError("dt.warmup_checkpoint must be an uncalibrated common DT warm-up checkpoint")
            if not resume: self.best_score=-math.inf  # best rollout of this branch
        else:
            norm=train_normalizer(ds)
            self.model.set_normalizer(norm)
            if self.ema_model is not None: self.ema_model.set_normalizer(norm)
        self.model.to(device); optimizer_to(self.optimizer,device)
        ema=None
        if self.ema_model is not None:
            self.ema_model.to(device)
            ema=hydra.utils.instantiate(cfg.ema,model=self.ema_model)
            ema.optimization_step=self.global_step
        log=SwanLabLogger(cfg,output,resume=resume)
        runner=None
        try:
            if train.enable_rollouts:
                runner=hydra.utils.instantiate(cfg.task.env_runner,output_dir=str(output))
                if hasattr(runner,"video_backend"): runner.video_backend="path"
            if self.rng is not None: restore(self.rng)
            while self.epoch<train.num_epochs:
                self.model.train()
                if self.epoch>=cfg.dt.warmup_epochs and self.dt_selected is None:
                    self.optimizer.zero_grad(set_to_none=True)
                    self._save("dt_warmup")
                    saved=capture()
                    try:
                        self.model.eval()
                        family,gain=calibrate(self.model,ds,cfg.dt,cfg.dataloader.batch_size,output,device,
                                              cfg.dataloader.num_workers)
                    finally:
                        self.model.train()
                        restore(saved)
                    self.model.set_gain(family,gain)
                    if self.ema_model is not None: self.ema_model.set_gain(family,gain)
                    self.dt_selected=family
                    self._save("latest")
                    print("DT locked for remaining epochs:",family,flush=True)
                phase=self.dt_selected or "warmup"
                totals={}
                samples=group_samples=0
                self.optimizer.zero_grad(set_to_none=True)
                with tqdm(total=batches,desc=f"DT-MP1 {phase} epoch {self.epoch+1}/{train.num_epochs}",
                          mininterval=train.tqdm_interval_sec,dynamic_ncols=True) as bar:
                    for i,batch in enumerate(loader):
                        batch=dict_apply(batch,lambda x:x.to(device,non_blocking=True))
                        loss,logs=self.model.compute_loss(batch)
                        n=batch["action"].shape[0]
                        # Scale before backward to avoid an inflated backward pass;
                        # tail/unequal batches are normalized at the group boundary.
                        expected=accum*cfg.dataloader.batch_size
                        (loss*(n/expected)).backward()
                        samples+=n; group_samples+=n
                        for k,v in logs.items(): totals[k]=totals.get(k,0.)+n*v
                        if (i+1)%accum==0 or i+1==batches:
                            if expected!=group_samples:
                                for p in self.model.parameters():
                                    if p.grad is not None: p.grad.mul_(expected/group_samples)
                            limit=train.max_grad_norm
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                float("inf") if limit is None else limit,error_if_nonfinite=True)
                            self.optimizer.step(); self.scheduler.step()
                            if ema is not None: ema.step(self.model)
                            self.optimizer.zero_grad(set_to_none=True)
                            group_samples=0; self.global_step+=1
                        bar.set_postfix(loss=f"{logs['bc_loss']:.4g}",refresh=False); bar.update()
                        if i+1==batches: break
                self.epoch+=1
                report={k:v/samples for k,v in totals.items()}
                report.update(epoch=self.epoch,global_step=self.global_step,dt_phase=phase)
                rollout={}
                if runner is not None and self.epoch%train.rollout_every==0:
                    policy=self.ema_model if self.ema_model is not None else self.model
                    policy.eval()
                    with torch.no_grad(): rollout=runner.run(policy)
                    report.update(scalar_metrics(rollout))
                score=next((report[k] for k in ("test/mean_score","test_mean_score","mean_score_0")
                            if k in report),None)
                if cfg.checkpoint.save_ckpt:
                    if score is not None and score>self.best_score and cfg.checkpoint.topk.k:
                        self.best_score=float(score); self._save("best")
                    if cfg.checkpoint.save_last_ckpt and (self.epoch%train.checkpoint_every==0 or self.epoch==train.num_epochs):
                        self._save("latest")
                log.log(report,step=self.global_step,rollout=rollout)
                with (output/"dt_metrics.jsonl").open("a",encoding="utf8") as f:
                    f.write(json.dumps(report,ensure_ascii=False)+"\n")
        finally:
            if runner is not None and hasattr(runner,"close"): runner.close()
            log.finish()
