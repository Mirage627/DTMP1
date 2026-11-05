"""DT CPU regression checks; these do not simulate CUDA or robot rollout success."""
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
from EfficientFlow.common.dt_mp1_calibration import train_normalizer, fit, partitions, choose, allocated_overhead
from EfficientFlow.common.dt_mp1_preflight import rollout_preflight
from EfficientFlow.common.mp1_observation_contract import validate_point_cloud
from EfficientFlow.common.point_cloud_provenance import check_generated_cloud, observation_provenance
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
    c.task.policy_shape_meta=OmegaConf.create({"obs":{"point_cloud":{"shape":[8,6],"type":"point_cloud"},
                                           "state":{"shape":[2]}},"action":{"shape":[10]}})
    c.policy.down_dims=[8,16]; c.policy.n_groups=2; c.policy.kernel_size=3
    c.policy.diffusion_step_embed_dim=8; c.policy.encoder_output_dim=8
    c.dt.warmup_epochs=1; c.dt.fit_batches=2
    c.dt.basis.rank=2; c.dt.basis.min_samples=2
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

    def test_zero_gain_adamw_updates_two_batch_sizes(self):
        for bs in (2,4):
            with self.subTest(batch_size=bs):
                dt,ref,b=self.models()
                b={"action":b["action"][:bs],"obs":{k:v[:bs] for k,v in b["obs"].items()}}
                losses=[]
                for m in (ref,dt):
                    opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=1e-6)
                    np.random.seed(9); torch.manual_seed(11)
                    loss,_=m.compute_loss(b); loss.backward(); losses.append(loss.detach())
                    torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True)
                    opt.step()
                torch.testing.assert_close(losses[0],losses[1],rtol=1e-5,atol=1e-6)
                for (name,p),(other,q) in zip(ref.named_parameters(),dt.named_parameters()):
                    self.assertEqual(name,other)
                    torch.testing.assert_close(p,q,rtol=4e-4,atol=4e-6)

    def test_gain_family_validation_is_atomic(self):
        dt,_,_=self.models(); k=len(dt.dt_K)
        previous=torch.eye(k)*.3
        dt.set_gain("diagonal",previous)
        bad=previous.clone(); bad[0,1]=.1
        unequal=previous.clone(); unequal[0,0]=.4
        cases=[("diagonal",bad),("sub_scalar",unequal),("full",torch.full((k,k),float('nan'))),
               ("global_scalar",torch.tensor(3.))]
        for family,gain in cases:
            with self.assertRaises(ValueError): dt.set_gain(family,gain)
            torch.testing.assert_close(dt.dt_K,previous)
        dt.set_gain("sub_scalar",previous)

    def test_actual_remaining_run_cost_allocation(self):
        self.assertAlmostEqual(allocated_overhead(100.,1000,2.,60.),3.)
        self.assertAlmostEqual(allocated_overhead(100.,10,2.,60.),100.)
        self.assertGreater(allocated_overhead(100.,500,2.,60.),allocated_overhead(100.,1000,2.,60.))
        with self.assertRaises(ValueError): allocated_overhead(1.,0,1.,60.)

    def test_point_cloud_generation_contract(self):
        self.assertIsNone(check_generated_cloud({}, required=False))
        with self.assertRaisesRegex(ValueError, 'missing'):
            check_generated_cloud({}, required=True)
        cloud=np.ones((1024,6),dtype=np.float32)
        report=check_generated_cloud({'point_cloud':cloud},required=True)
        self.assertEqual(report['max'],[1.]*6)
        cloud[0,4]=np.inf
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            check_generated_cloud({'point_cloud':cloud})
        provenance=observation_provenance()
        self.assertIn('robomimic.envs.env_robosuite',provenance['modules'])
        self.assertFalse(provenance['coordinate_frame_verified'])

    def test_observation_contract_and_preflight(self):
        dt,_,b=self.models()
        raw={k:v.numpy() for k,v in b['obs'].items()}
        steps=[]
        env=SimpleNamespace(reset=lambda:raw,step=lambda a:(steps.append(a) or (raw,None,None,None)))
        runner=SimpleNamespace(env=env,abs_action=False)
        with tempfile.TemporaryDirectory() as d:
            report=rollout_preflight(runner,dt,d,(8,6))
            self.assertFalse(report['coordinate_frame_verified'])
            self.assertFalse(report['color_scale_verified'])
            self.assertEqual(steps[0].shape,(4,3,10))
            self.assertTrue(dt.training)
            with patch.object(env,'reset',return_value={'state':raw['state']}):
                with self.assertRaisesRegex(RuntimeError,'point cloud'): rollout_preflight(runner,dt,d,(8,6))
            self.assertTrue(dt.training)
        cloud=raw['point_cloud'][0,0].copy()
        validate_point_cloud({'point_cloud':cloud},'point_cloud',(8,6))
        with self.assertRaisesRegex(ValueError,'missing'):
            validate_point_cloud({},'point_cloud',(8,6))
        cloud[0,0]=np.nan
        with self.assertRaisesRegex(ValueError,'non-finite'):
            validate_point_cloud({'point_cloud':cloud},'point_cloud',(8,6))

    def test_coffee_shapes_active_dt_vs_explicit_target(self):
        import hydra
        from EfficientFlow.model.common.normalizer import LinearNormalizer
        from EfficientFlow.policy.mp1 import adaptive_l2_loss
        register_resolvers()
        with initialize_config_dir(config_dir=str(ROOT/"config"),version_base=None):
            cfg=compose(config_name="DT_MP1",overrides=["task_name=coffee_preparation_d1","n_demo=1000"])
        self.assertFalse(any(s.get('type')=='rgb' for s in cfg.task.dataset.shape_meta.obs.values()))
        self.assertIn('agentview_image',cfg.task.env_runner.shape_meta.obs)
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
        keys={"global_scalar:1":"global_scalar","diagonal:1":"diagonal","strong":"global_scalar:1"}
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
            self.assertEqual(rep['remaining_training_batches'],1)
            self.assertIn('cost_scope',rep)
            self.assertIn('dct_all:r10',rep['basis_fit'])
            self.assertIn('joint_gradient:r2',rep['basis_fit'])
            for candidate in rep['fitted_candidates'].values():
                if candidate['family'] not in ('baseline','global_scalar'):
                    self.assertEqual(len(candidate['gain']),len(candidate['basis'][0]))
            data=json.loads((Path(d)/"dt_data_split.json").read_text())
            self.assertEqual(data['network_train_episode_count'],8)
            self.assertEqual(data['heldout_episode_count'],8)
            metrics=[json.loads(line) for line in (Path(d)/'dt_metrics.jsonl').read_text().splitlines()]
            for row in metrics:
                self.assertTrue(0<=row['grad_clip_fraction']<=1)
                self.assertTrue(0<row['grad_clip_scale']<=1)
                self.assertIn('dt_off_diagonal_fraction',row)
            for tag in ('dt_early','dt_middle','dt_late'):
                self.assertTrue((Path(d)/f'checkpoints/{tag}.ckpt').is_file())
            from EfficientFlow.common.dt_mp1_lifetime import diagnose
            before={tag:(Path(d)/f'checkpoints/{tag}.ckpt').read_bytes()
                    for tag in ('dt_early','dt_middle','dt_late')}
            life=diagnose([Path(d)/f'checkpoints/{tag}.ckpt' for tag in before],
                          Path(d)/'lifetime',device='cpu',pairs=2)
            self.assertIn('early_gain',life['stages']['middle']['summary'])
            self.assertIn('middle_gain',life['stages']['late']['summary'])
            self.assertFalse(life['training_gain_modified'])
            for tag,content in before.items():
                self.assertEqual((Path(d)/f'checkpoints/{tag}.ckpt').read_bytes(),content)
            resumed=TrainDTMP1Workspace(cfg,d); resumed.run()
            self.assertEqual(resumed.dt_selected,ws.dt_selected)
            torch.testing.assert_close(resumed.model.dt_K,ws.model.dt_K)
            for a,b in zip(ws.model.parameters(),resumed.model.parameters()): torch.testing.assert_close(a,b)
            cfg.dt.warmup_checkpoint=str(Path(d)/"checkpoints/dt_warmup.ckpt")
            cfg.dt.family="baseline"
            fork=TrainDTMP1Workspace(cfg,Path(d)/"fork"); fork.run()
            self.assertEqual((fork.dt_selected,fork.global_step),("baseline",2))

    def test_preflight_only_exits_without_training_and_closes_runner(self):
        cfg=config(); ds=Data()
        cfg.dt.preflight_only=True
        cfg.training.enable_rollouts=True
        cfg.task.dataset._target_='tests.SyntheticReplayForDT'
        cfg.task.env_runner._target_='tests.SyntheticRunnerForDT'
        batch=next(iter(torch.utils.data.DataLoader(ds,batch_size=2)))
        obs={k:v.numpy() for k,v in batch['obs'].items()}
        closed=[]
        runner=SimpleNamespace(env=SimpleNamespace(reset=lambda:obs,step=lambda a:(obs,None,None,None)),
                               abs_action=False,close=lambda:closed.append(True))
        import hydra
        original=hydra.utils.instantiate
        def build(node,*args,**kw):
            target=node.get('_target_','')
            if target=='tests.SyntheticReplayForDT': return ds
            if target=='tests.SyntheticRunnerForDT': return runner
            return original(node,*args,**kw)
        with tempfile.TemporaryDirectory() as d, patch('hydra.utils.instantiate',side_effect=build):
            ws=TrainDTMP1Workspace(cfg,d); ws.run()
            self.assertEqual((ws.epoch,ws.global_step),(0,0))
            self.assertTrue((Path(d)/'dt_rollout_preflight.json').is_file())
            self.assertFalse((Path(d)/'checkpoints').exists())
            self.assertEqual(closed,[True])

    def test_baseline_alias_uses_identical_paired_costs(self):
        from EfficientFlow.common.dt_mp1_calibration import evaluate
        dt,_,_=self.models()
        scores,_=evaluate(dt,Data(),{'baseline':('baseline',None),'early_gain':('baseline',None)},
                          2,2,101,torch.device('cpu'),'test')
        self.assertEqual(scores['baseline'],scores['early_gain'])

    def test_basis_vjp_and_proxy_gradient_against_linear_analysis(self):
        from EfficientFlow.common.dt_mp1_basis import solve_basis
        dt,_,_=self.models()
        torch.manual_seed(27)
        A=torch.randn(40,40,dtype=torch.float64)*.1
        x=torch.randn(5,4,10,dtype=torch.float64)
        v=torch.randn_like(x)
        t=torch.tensor([.2,.4,.6,.8,1.],dtype=torch.float64)
        r=torch.tensor([.2,.1,.2,.3,0.],dtype=torch.float64)
        class Linear(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.weight=torch.nn.Parameter(A.clone())
            def forward(self,x,t,r,global_cond,training=False):
                return (x.flatten(1)@self.weight.T).reshape_as(x)
        dt.model=Linear()
        with patch.object(dt,'_draw',return_value=(x,t,r,torch.zeros(5,2),v,v)):
            (vv,ell,hte),delta=dt.basis_rows({})
        active=t>r
        H=(t-r)[active,None,None]*A
        expected_v=v[active].flatten(1)
        expected_ell=expected_v-torch.einsum('nij,nj->ni',H,expected_v)
        torch.testing.assert_close(vv,expected_v)
        torch.testing.assert_close(ell,expected_ell)
        torch.testing.assert_close(hte,torch.einsum('nji,nj->ni',H,expected_ell))
        self.assertIsNone(dt.model.weight.grad)
        M=hte.T@vv/len(vv)
        direction=torch.randn_like(A)
        def objective(B):
            residual=ell-torch.einsum('nij,jk,nk->ni',H,B,vv)
            return residual.square().sum(1).mean()+.3*B.square().sum()
        eps=1e-6
        finite_difference=(objective(eps*direction)-objective(-eps*direction))/(2*eps)
        torch.testing.assert_close(finite_difference,-2*(M*direction).sum(),rtol=1e-7,atol=1e-7)
        # Pure skew moments are a real limitation of the symmetric heuristic.
        skew=torch.tensor([[0.,1.],[-1.,0.]])
        u,info=solve_basis(skew,1)
        self.assertIsNone(u)
        self.assertIn('spectrum',info['reason'])
        skew3=torch.zeros(3,3); skew3[:2,:2]=skew
        u,_=solve_basis(skew3,2,mode='joint_gradient')
        torch.testing.assert_close(u@u.T,torch.diag(torch.tensor([1.,1.,0.])))

    def test_basis_directions_all_channels_and_spectrum_fallback(self):
        from EfficientFlow.policy.dt_mp1 import action_basis
        from EfficientFlow.common.dt_mp1_basis import solve_basis, estimate_basis
        u=action_basis(4,10,rank=10)
        torch.testing.assert_close(u.T@u,torch.eye(10))
        self.assertTrue((u.reshape(4,10,10).square().sum((0,2))>0).all())
        with self.assertRaises(ValueError): action_basis(4,10,rank=8)
        score=torch.diag(torch.arange(1.,41.)); score[39,39]=-100
        u,_=solve_basis(score,2)
        self.assertAlmostEqual(float((u@u.T)[39,39]),1.)
        self.assertAlmostEqual(float((u@u.T)[38,38]),1.)
        for bad in (torch.zeros(40,40),torch.eye(40),torch.full((40,40),float('nan'))):
            self.assertIsNone(solve_basis(bad,2)[0])
        dt,_,batch=self.models(); cfg=config().dt.basis
        cfg.min_samples=1000
        u,info=estimate_basis(dt,[batch],cfg,torch.device('cpu'))
        self.assertIsNone(u)
        self.assertEqual(sum(info['delta_bins']),4)

    def test_correction_state_dynamic_rank_integrity_and_ema(self):
        from EfficientFlow.common.dt_mp1_state import Correction, apply_correction, normalizer_fingerprint
        from EfficientFlow.model.diffusion.ema_model import EMAModel
        dt,_,_=self.models(); ema=copy.deepcopy(dt)
        U=torch.linalg.qr(torch.randn(40,3)).Q
        K=torch.tensor([[.2,.3,0.],[-.1,0.,0.],[0.,.1,-.3]])
        correction=Correction(U,'full',K,{'normalizer_sha256':normalizer_fingerprint(dt)})
        apply_correction(correction,dt,ema)
        with self.assertRaisesRegex(ValueError,'old K'): dt.set_basis(torch.flip(U,[0]))
        state=dt.state_dict()
        restored,_,_=self.models()
        restored.load_state_dict(state)
        self.assertEqual(tuple(restored.dt_K.shape),(3,3))
        torch.testing.assert_close(restored.dt_U,U)
        torch.testing.assert_close(restored.dt_K,K)
        for key in ('dt_U','dt_K','normalizer.params_dict.action.scale'):
            with self.subTest(field=key):
                damaged=copy.deepcopy(state); damaged[key]=damaged[key]+.01
                with self.assertRaisesRegex(ValueError,'fingerprint'): restored.load_state_dict(damaged)
        before=dt.dt_U.clone()
        bad=Correction(torch.zeros(40,3),'full',K)
        with self.assertRaises(ValueError): apply_correction(bad,dt,ema)
        torch.testing.assert_close(dt.dt_U,before)
        wrong=Correction(U,'full',K,{'normalizer_sha256':'wrong'})
        with self.assertRaisesRegex(ValueError,'normalizer'): apply_correction(wrong,dt,ema)
        EMAModel(ema).step(dt)
        torch.testing.assert_close(ema.dt_U,dt.dt_U)
        torch.testing.assert_close(ema.dt_K,dt.dt_K)
        self.assertFalse(dt.dt_U.requires_grad or dt.dt_K.requires_grad)
        altered=copy.deepcopy(dt.normalizer)
        with torch.no_grad(): altered.params_dict['action']['scale'].add_(.01)
        with self.assertRaisesRegex(ValueError,'normalizer'): dt.set_normalizer(altered)
        broken_ema=copy.deepcopy(ema)
        broken_ema.normalizer.load_state_dict(altered.state_dict())
        changed=Correction(torch.flip(U,[0]),'full',K,correction.metadata)
        with self.assertRaisesRegex(ValueError,'normalizer'): apply_correction(changed,dt,broken_ema)
        torch.testing.assert_close(dt.dt_U,U)
        # Explicit migration of actual v3 fixed-basis state, with no v4 checksum.
        from EfficientFlow.policy.dt_mp1 import action_basis
        dt.set_gain('baseline'); dt.set_basis(action_basis(4,10,'legacy_first4'))
        old=dt.state_dict(); del old['dt_family']; del old['dt_integrity']
        restored.load_state_dict(old)
        self.assertEqual(restored.dt_U.shape[1],8)

    def test_whole_correction_transfer_restores_state_and_family_aliases(self):
        from EfficientFlow.common.dt_mp1_state import Correction
        from EfficientFlow.common.dt_mp1_calibration import evaluate
        dt,_,_=self.models()
        original=Correction.capture(dt)
        U=torch.linalg.qr(torch.randn(40,3)).Q
        options={'baseline':original,'middle_gain':Correction(U,'full',torch.eye(3)*.1)}
        scores,_=evaluate(dt,Data(),options,2,2,101,torch.device('cpu'),'transfer')
        self.assertIn('middle_gain',scores)
        torch.testing.assert_close(dt.dt_U,original.basis)
        s=lambda v:np.full(6,v,dtype=float)
        risks={'baseline':(s(4),s(0),1),'early_gain':(s(2),s(0),1),'middle_gain':(s(3),s(0),1)}
        selected,_=choose({'early_gain':'global_scalar','middle_gain':'full','strong':'early_gain'},risks,.05)
        self.assertEqual(selected,'early_gain')

    def test_relative_tail_rejection_and_frozen_basis(self):
        from EfficientFlow.common.dt_mp1_state import Correction
        from EfficientFlow.common.dt_mp1_calibration import relative_safety
        dt,_,_=self.models(); cfg=config().dt
        base=Correction.capture(dt)
        candidate=Correction(base.basis,'global_scalar',torch.tensor(.5))
        options={'baseline':base,'candidate':candidate}
        with patch.object(dt,'correction_ratios',return_value=(torch.tensor([3.,4.]),torch.tensor([.1,.9]))):
            retained,report=relative_safety(dt,Data(),options,cfg,2,4,torch.device('cpu'),0)
        self.assertEqual(list(retained),['baseline'])
        self.assertFalse(report['candidate']['accepted'])
        torch.testing.assert_close(dt.dt_U,base.basis)

    def test_global_scalar_fit_does_not_depend_on_basis_responses(self):
        torch.manual_seed(5)
        hv=torch.randn(12,40); ell=.5*hv
        model=SimpleNamespace(dt_K=torch.zeros(3,3),
            calibration_rows=lambda _,need_basis:(torch.empty(12,40,0),torch.zeros(12,3),hv,ell))
        options,n=fit(model,[{'action':torch.zeros(1)}],['global_scalar'],'cpu')
        self.assertEqual(n,12)
        torch.testing.assert_close(options['global_scalar:0.1'][1],torch.tensor(.5/1.1))

    def test_additional_review_skew_example_and_curvature(self):
        from EfficientFlow.common.dt_mp1_basis import solve_basis
        H=torch.tensor([[.5,-.5],[.5,.5]],dtype=torch.float64)
        K=torch.tensor([[0.,1.],[-1.,0.]],dtype=torch.float64)
        eye=torch.eye(2,dtype=torch.float64)
        M=H.T@(eye-H)
        torch.testing.assert_close(M,torch.tensor([[0.,.5],[-.5,0.]],dtype=torch.float64))
        torch.testing.assert_close(H@K,eye-H)
        self.assertAlmostEqual(float((eye-H).square().sum()),1.)
        self.assertIsNone(solve_basis(M,2,mode='gradient')[0])
        U,_=solve_basis(M,2,mode='joint_gradient')
        U=U.double()
        transformed=U.T@K@U
        torch.testing.assert_close(H@U@transformed@U.T,eye-H)
        # Feasible proxy example: the larger first-order score loses after curvature.
        H=torch.diag(torch.tensor([2.,.4],dtype=torch.float64))
        covariance=torch.diag(torch.tensor([.1,.8],dtype=torch.float64))
        f=(H.T@(eye-H)@covariance).diagonal()
        curvature=(H.T@H@covariance).diagonal()
        eta=.01
        optimal=f/(curvature+eta)
        improvement=f.square()/(curvature+eta)
        self.assertGreater(abs(f[0]),abs(f[1]))
        self.assertLess(improvement[0],improvement[1])
        self.assertTrue((optimal.abs()<2).all())
        for j in range(2):
            before=((eye-H)@covariance@(eye-H).T).trace()
            B=torch.zeros_like(H); B[j,j]=optimal[j]
            residual=eye-H-H@B
            after=(residual@covariance@residual.T).trace()+eta*B.square().sum()
            torch.testing.assert_close(before-after,improvement[j])

    def test_zero_full_moment_is_optimal_for_one_global_gain(self):
        h=torch.tensor([.5,(1+2**.5)/2],dtype=torch.float64)
        self.assertAlmostEqual(float((h*(1-h)).mean()),0.)
        baseline=((1-h).square()).mean()
        for gain in (-2.,-.5,0.,.5,2.):
            risk=((1-h-h*gain).square()).mean()+.01*gain**2
            self.assertGreaterEqual(float(risk),float(baseline)-1e-12)
        # Different gains per bucket enlarge the model class and can do better.
        torch.testing.assert_close(1-h-h*((1-h)/h),torch.zeros_like(h))

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
