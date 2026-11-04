"""DT v2.1 CPU checks: real six-channel MP1 policies, sampler contracts, and trainer."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent))
from EfficientFlow.common.train_entry import register_resolvers
from EfficientFlow.policy.mp1 import Meanpolicy
from EfficientFlow.policy.dt_mp1 import DTMP1Policy
from EfficientFlow.common.dt_mp1_calibration import train_normalizer, fit, partitions, choose
from EfficientFlow.workspace.train_dt_mp1_workspace import TrainDTMP1Workspace, preflight_hdf5


class Replay(dict): pass


class Data(torch.utils.data.Dataset):
    def __init__(self):
        rng=np.random.default_rng(33)
        self.replay_buffer=Replay()
        self.replay_buffer.episode_ends=np.arange(4,65,4)
        self.train_mask=np.arange(16)<8
        self.pc_keys=["point_cloud"]; self.lowdim_keys=["state"]
        for k,shape in (("point_cloud",(64,8,6)),("state",(64,2)),("action",(64,10))):
            self.replay_buffer[k]=rng.normal(size=shape).astype("f")
        self.replay_buffer["state"][:,1]=.3
        self.replay_buffer["state"][32:,0]+=20   # must not leak into the fit
        self.sampler=SimpleNamespace(indices=np.array([[i*4] for i in range(8)]))

    def get_validation_dataset(self):
        held=copy.copy(self)
        held.sampler=SimpleNamespace(indices=np.array([[i*4] for i in range(8,16)]))
        return held

    def __len__(self): return len(self.sampler.indices)

    def __getitem__(self,i):
        a=self.sampler.indices[i,0]
        buf=self.replay_buffer
        return {"action":torch.tensor(buf["action"][a:a+4]),
                "obs":{k:torch.tensor(buf[k][a:a+2]) for k in self.lowdim_keys+self.pc_keys}}


def config():
    register_resolvers()
    with initialize_config_dir(config_dir=str(ROOT/"config"),version_base=None):
        c=compose(config_name="DT_MP1",overrides=["task_name=coffee_preparation_d1","n_demo=1000"])
    c.task.shape_meta=OmegaConf.create({"obs":{"point_cloud":{"shape":[8,6],"type":"point_cloud"},
                                           "state":{"shape":[2]}},"action":{"shape":[10]}})
    c.policy.down_dims=[8,16]; c.policy.n_groups=2; c.policy.kernel_size=3
    c.policy.diffusion_step_embed_dim=8; c.policy.encoder_output_dim=8
    c.dt.warmup_epochs=1; c.dt.fit_batches=2
    c.dt.tune_pairs=c.dt.accept_pairs=c.dt.report_pairs=2
    c.dataloader.batch_size=2; c.dataloader.num_workers=0
    c.training.device="cpu"; c.training.num_epochs=2; c.training.max_train_steps=1
    c.training.enable_rollouts=False; c.logging.mode="disabled"
    return c


class DTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def models(self):
        cfg=config()
        import hydra
        dt=hydra.utils.instantiate(cfg.policy)
        cfg.policy._target_="EfficientFlow.policy.mp1.Meanpolicy"
        ref=hydra.utils.instantiate(cfg.policy)
        ref.load_state_dict({k:v for k,v in dt.state_dict().items() if not k.startswith("dt_")})
        ds=Data(); norm=train_normalizer(ds)
        for m in (ref,dt): m.set_normalizer(norm)
        batch=next(iter(torch.utils.data.DataLoader(ds,batch_size=4)))
        return dt,ref,batch

    def test_mp1_zero_gain_full_gradient_and_inference(self):
        dt,ref,b=self.models()
        for m in (ref,dt):
            np.random.seed(9); torch.manual_seed(11)
            loss,_=m.compute_loss(b); loss.backward()
        for (n,p),(n2,q) in zip(ref.named_parameters(),dt.named_parameters()):
            self.assertEqual(n,n2)
            if p.grad is not None: torch.testing.assert_close(q.grad,p.grad,rtol=4e-4,atol=4e-6)
        noise=torch.randn_like(b["action"])
        dt.set_gain("diagonal",torch.eye(dt.dt_K.shape[0])*.4)
        a=dt.predict_action(b["obs"],initial_noise=noise)
        rb=ref.predict_action(b["obs"],initial_noise=noise)
        torch.testing.assert_close(a["action_pred"],rb["action_pred"])
        self.assertEqual(tuple(a["action"].shape),(4,3,10))
        torch.testing.assert_close(a["action"],a["action_pred"][:,1:4])

    def test_cached_label_non_symmetric_and_diagonal_times(self):
        dt,_,b=self.models()
        np.random.seed(14); torch.manual_seed(19)
        x,t,r,c,d,v=dt._draw(b)
        U=dt.dt_U; k=U.shape[1]; mask=t>r
        torch.testing.assert_close(U.T@U,torch.eye(k))
        R=torch.stack([dt._jvp(x,t,r,c,u.reshape(1,4,10).expand_as(x),0).flatten(1)
                       for u in U.T],-1)*(t-r)[:,None,None]
        K=torch.zeros(k,k); K[2,1]=.7; K[0,3]=-.4
        dt.set_gain("full",K)
        tang=dt.tangent(d,v,mask)
        torch.testing.assert_close(tang[~mask],d[~mask])
        direct=(t-r)[:,None]*(dt._jvp(x,t,r,c,tang,1)-dt._jvp(x,t,r,c,d,1)).flatten(1)
        correct=R@((v.flatten(1)@U)@K.T).unsqueeze(-1)
        torch.testing.assert_close(direct,correct.squeeze(-1),rtol=2e-4,atol=2e-6)
        self.assertGreater(direct.norm().item(),1e-6)

    def test_coffee_shapes_active_dt_vs_explicit_target(self):
        import hydra
        from EfficientFlow.model.common.normalizer import LinearNormalizer
        from EfficientFlow.policy.mp1 import adaptive_l2_loss
        register_resolvers()
        with initialize_config_dir(config_dir=str(ROOT/"config"),version_base=None):
            cfg=compose(config_name="DT_MP1",overrides=["task_name=coffee_preparation_d1","n_demo=1000"])
        cfg.policy.down_dims=[8,16]; cfg.policy.n_groups=2
        cfg.policy.diffusion_step_embed_dim=8; cfg.policy.encoder_output_dim=8
        model=hydra.utils.instantiate(cfg.policy)
        torch.manual_seed(19)
        a=torch.randn(4,4,10)
        obs={k:torch.randn(4,2,*s["shape"]) for k,s in cfg.task.env_runner.shape_meta.obs.items()}
        norm=LinearNormalizer(); norm.fit({"action":a,**{k:v for k,v in obs.items() if "image" not in k}})
        model.set_normalizer(norm); batch={"obs":obs,"action":a}
        # Independent direct construction with actual task widths and both RGB cameras.
        gain=torch.linspace(-.4,.6,model.dt_K.shape[0]); model.set_gain("diagonal",torch.diag(gain))
        np.random.seed(15); torch.manual_seed(9)
        x,t,r,c,d,v=model._draw(batch)
        change=((v.flatten(1)@model.dt_U)*gain)@model.dt_U.T
        tangent=d+torch.where((t>r)[:,None,None],change.reshape_as(d),torch.zeros_like(d))
        target=d-(t-r)[:,None,None]*model._jvp(x,t,r,c,tangent,1)
        pred,features=model.model(x,t,r=r,global_cond=c)
        expected=adaptive_l2_loss(pred-target.detach())+.5*sum(model.dispersive_loss(f) for f in features)
        np.random.seed(15); torch.manual_seed(9)
        actual,_=model.compute_loss(batch)
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
        actual.backward()
        self.assertTrue(torch.isfinite(next(model.obs_encoder.parameters()).grad).all())
        actions=model.predict_action(obs,initial_noise=torch.randn_like(a))
        self.assertEqual(tuple(actions["action"].shape),(4,3,10))

    def test_fit_all_families_and_gate(self):
        torch.manual_seed(5)
        R=torch.randn(12,4,2); p=torch.randn(12,2); hv=torch.randn(12,4)
        ell=torch.einsum("ndk,kj,nj->nd",R,torch.tensor([[.1,.5],[-.6,.2]]),p)
        model=SimpleNamespace(dt_K=torch.zeros(2,2),calibration_rows=lambda _: (R,p,hv,ell))
        opts,n=fit(model,[{"obs":torch.zeros(1)}],["global_scalar","sub_scalar","diagonal","full"],"cpu")
        self.assertEqual(n,12)
        design=(R.double().unsqueeze(-1)*p.double()[:,None,None,:]).reshape(-1,4)
        C=design.T@design/12
        expect=torch.linalg.solve(C+.1*C.diag().mean()*torch.eye(4,dtype=C.dtype),
                                  design.T@ell.double().flatten()/12)
        torch.testing.assert_close(opts["full:0.1"][1],expect.reshape(2,2).float())
        keys={"global_scalar:1":None,"diagonal:1":None,"strong":"global_scalar:1"}
        s=lambda v:np.full(6,v,dtype=float)
        risks={"baseline":(s(4),s(0),1),"global_scalar:1":(s(3),s(.2),1),
               "diagonal:1":(s(1),s(.1),1)}
        self.assertEqual(choose(keys,risks,.05)[0],"diagonal:1")
        risks["diagonal:1"]=(s(1),s(5),1)
        self.assertEqual(choose(keys,risks,.05)[0],"global_scalar:1")

    def test_train_only_stats_and_trajectory_splits(self):
        ds=Data(); norm=train_normalizer(ds)
        self.assertLess(norm["state"].get_input_stats()["max"][0].item(),10)
        torch.testing.assert_close(norm["state"].normalize(torch.tensor([[0.,.3]]))[0,1],torch.tensor(0.))
        _,ids=partitions(ds,33)
        self.assertEqual(sorted(sum(ids.values(),[])),list(range(8,16)))

    def test_train_calibrate_resume_and_common_warmup_fork(self):
        cfg=config(); ds=Data()
        # Use our synthetic replay loader for this lifecycle check. HDF5
        # ingestion contracts are exercised independently below.
        cfg.task.dataset._target_="tests.SyntheticReplayForDT"
        import hydra
        original=hydra.utils.instantiate
        def build(node,*args,**kw):
            if node.get("_target_","")=="tests.SyntheticReplayForDT": return ds
            return original(node,*args,**kw)
        with tempfile.TemporaryDirectory() as d, patch("hydra.utils.instantiate",side_effect=build):
            ws=TrainDTMP1Workspace(cfg,d); ws.run()
            self.assertEqual((ws.epoch,ws.global_step),(2,2))
            rep=json.loads((Path(d)/"dt_calibration.json").read_text())
            self.assertIn(rep["selected"],rep["candidates"])
            resumed=TrainDTMP1Workspace(cfg,d); resumed.run()
            self.assertEqual(resumed.dt_selected,ws.dt_selected)
            torch.testing.assert_close(resumed.model.dt_K,ws.model.dt_K)
            for a,b in zip(ws.model.parameters(),resumed.model.parameters()): torch.testing.assert_close(a,b)
            cfg.dt.warmup_checkpoint=str(Path(d)/"checkpoints/dt_warmup.ckpt")
            cfg.dt.family="baseline"
            fork=TrainDTMP1Workspace(cfg,Path(d)/"fork"); fork.run()
            self.assertEqual((fork.dt_selected,fork.global_step),("baseline",2))

    def test_real_hdf5_layout_preflight(self):
        import h5py
        cfg=config()
        cfg.n_demo=1
        with tempfile.TemporaryDirectory() as d:
            cfg.dataset_path=str(Path(d)/"coffee.hdf5")
            with h5py.File(cfg.dataset_path,"w") as f:
                demo=f.create_group("data/demo_0")
                demo.create_dataset("actions",(4,7),dtype="f")
                obs=demo.create_group("obs")
                for key,spec in cfg.task.dataset.shape_meta.obs.items():
                    shape=tuple(spec["shape"])
                    if spec.get("type")=="rgb": shape=(shape[1],shape[2],shape[0])
                    obs.create_dataset(key,(4,*shape),dtype="f")
            preflight_hdf5(cfg)
            with h5py.File(cfg.dataset_path,"a") as f:
                obs=f["data/demo_0/obs"]
                del obs["point_cloud"]
                obs.create_dataset("point_cloud",(4,1024,3),dtype="f")
            with self.assertRaisesRegex(ValueError,"point_cloud"): preflight_hdf5(cfg)


if __name__=="__main__": unittest.main()
