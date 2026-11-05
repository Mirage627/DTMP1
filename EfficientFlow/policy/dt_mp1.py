"""MP1 spatial-tangent calibration with explicit gain-family contracts."""
import math
import torch
from EfficientFlow.policy.mp1 import Meanpolicy, adaptive_l2_loss
from EfficientFlow.common.pytorch_util import dict_apply


def action_basis(horizon, action_dim):
    # Rows match tensor.flatten(1): time first, action coordinate last.
    basis = []
    for ch in range(min(4, action_dim)):
        for freq in range(min(2, horizon)):
            b = torch.zeros(horizon, action_dim, dtype=torch.float64)
            t = torch.arange(horizon, dtype=b.dtype) + .5
            col = torch.cos(freq * math.pi * t / horizon)
            b[:, ch] = col / col.norm()
            basis.append(b.flatten())
    return torch.stack(basis, dim=1).float()


class DTMP1Policy(Meanpolicy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.obs_as_global_cond or self.condition_type != "film":
            raise ValueError("DT-MP1 requires MP1 global FiLM conditioning")
        if not self.use_pc_color or kwargs["shape_meta"]["obs"]["point_cloud"]["shape"][-1] != 6:
            raise ValueError("DT-MP1 expects normalized XYZRGB with six channels")
        if not 1 <= self.n_action_steps <= self.horizon-self.n_obs_steps+1:
            raise ValueError("Action execution window extends beyond the predicted horizon")
        u = action_basis(self.horizon, self.action_dim)
        self.register_buffer("dt_U", u)
        self.register_buffer("dt_K", torch.zeros(u.shape[1], u.shape[1]))
        self.register_buffer("dt_global", torch.zeros(()))

    @torch.no_grad()
    def set_gain(self, family, gain=None):
        k = torch.zeros_like(self.dt_K)
        g = torch.zeros_like(self.dt_global)
        if family != "baseline":
            value = torch.as_tensor(gain, device=k.device, dtype=k.dtype)
            if not torch.isfinite(value).all():
                raise ValueError("Non-finite DT gain")
            if family == "global_scalar" and value.ndim == 0:
                g.copy_(value)
                size = g.abs()
            elif family in ("sub_scalar", "diagonal", "full") and value.shape == k.shape:
                diagonal = torch.diag(torch.diagonal(value))
                if family in ("diagonal", "sub_scalar") and not torch.equal(value, diagonal):
                    raise ValueError(f"{family} requires an exactly diagonal gain")
                if family == "sub_scalar" and not torch.equal(
                        value, torch.eye(len(k), device=k.device, dtype=k.dtype)*value[0, 0]):
                    raise ValueError("sub_scalar requires a scalar multiple of identity")
                k.copy_(value)
                size = torch.linalg.matrix_norm(k, ord=2)
            else:
                raise ValueError(f"Wrong DT gain shape for {family}: {value.shape}")
            if not torch.isfinite(size) or size > 2:
                raise ValueError("Invalid DT gain")
        self.dt_K.copy_(k)
        self.dt_global.copy_(g)

    def _normalize_observations(self, observations):
        pc = observations["point_cloud"]
        if pc.ndim != 4 or pc.shape[-1] != 6 or pc.shape[1] < self.n_obs_steps:
            raise ValueError(f"Expected [B,To,N,6] XYZRGB; received {tuple(pc.shape)}")
        for key in ("point_cloud", *self.obs_encoder.state_keys):
            if key not in self.normalizer.params_dict:
                raise ValueError(f"MP1 normalizer has no fitted field: {key}")
            if self.normalizer[key].params_dict["scale"].numel() != observations[key].shape[-1]:
                raise ValueError(f"Normalizer channel mismatch: {key}")
        return super()._normalize_observations(observations)

    def _draw(self, batch):
        a = batch["action"]
        if a.ndim != 3 or a.shape[1:] != (self.horizon, self.action_dim):
            raise ValueError(f"Expected [B,{self.horizon},{self.action_dim}] actions; got {a.shape}")
        if self.normalizer["action"].params_dict["scale"].numel() != self.action_dim:
            raise ValueError("Action normalizer width differs from the model")
        a = self.normalizer["action"].normalize(a)
        obs = self._normalize_observations(batch["obs"])
        inputs = dict_apply(obs, lambda n: n[:, :self.n_obs_steps].reshape(-1,*n.shape[2:]))
        c = self.obs_encoder(inputs).reshape(a.shape[0], -1)
        t,r = self.sample_t_r(a.shape[0], a.device)
        z = torch.randn_like(a)
        x = (1-t[:,None,None])*a+t[:,None,None]*z
        with torch.no_grad():
            m = self.model(x,t,r=t,global_cond=c.detach(),training=False)
            dw = self.w*(z-a)+(1-self.w)*m
        return x,t,r,c,dw,dw-m

    def tangent(self, dw, v, active):
        if not (self.dt_global.ne(0) | self.dt_K.ne(0).any()).item():
            return dw
        flat = v.flatten(1)
        change = self.dt_global*flat + ((flat @ self.dt_U) @ self.dt_K.T) @ self.dt_U.T
        return torch.where(active[:,None,None], dw+change.reshape_as(dw), dw)

    def _jvp(self, x,t,r,c,v, temporal):
        def velocity(xx,tt,rr):
            return self.model(xx,tt,r=rr,global_cond=c.detach(),training=False)
        return torch.autograd.functional.jvp(velocity, (x,t,r),
            (v, torch.full_like(t, temporal), torch.zeros_like(r)), create_graph=False)[1]

    def compute_loss(self, batch, *, diagnostic="normal"):
        if diagnostic not in ("normal","time","none"):
            raise ValueError(diagnostic)
        with torch.enable_grad():
            x,t,r,c,dw,v = self._draw(batch)
            active = t > r
            tangent = dw
            if diagnostic != "none":
                tangent = self.tangent(dw,v,active) if diagnostic=="normal" else torch.zeros_like(dw)
                # Keep the original MP1 full-batch primal/JVP graph for every
                # family. Re-running the primal separately and masking JVP rows
                # can perturb near-zero gradients; AdamW can amplify those ULPs.
                (pred,features),(j,_) = torch.autograd.functional.jvp(
                    lambda xx,tt,rr:self.model(xx,tt,r=rr,global_cond=c),
                    (x,t,r),(tangent,torch.ones_like(t),torch.zeros_like(r)),create_graph=True)
                y = dw-(t-r)[:,None,None]*j
            else:
                pred,features = self.model(x,t,r=r,global_cond=c)
                y = dw
            error = pred-y.detach()
            reg = adaptive_l2_loss(error)
            dis = error.new_zeros(())
            for f in features:
                dist = torch.cdist(f,f,p=2).square()
                peak = dist.max()
                # The original is undefined for an identical-feature batch.
                safe_peak = torch.where(peak>0, peak, torch.ones_like(peak))
                dis = dis+torch.exp(-dist/safe_peak).mean().log()
            loss = reg+.5*dis
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite DT-MP1 loss; check input data/normalizer")
            return loss, {"bc_loss":loss.item(), "meanflow_loss":reg.item(),
                          "dis_loss":dis.item(), "mse_val":error.detach().square().mean().item(),
                          "dt_off_diagonal_fraction":active.float().mean().item(),
                          "dt_correction_norm":(tangent-dw).flatten(1).norm(dim=1).mean().item(),
                          "dt_gain_nonzero":float(bool(self.dt_global.ne(0) | self.dt_K.ne(0).any()))}

    @torch.no_grad()
    def calibration_rows(self, batch):
        x,t,r,c,_,v = self._draw(batch)
        active = t > r
        if not active.any():
            return None
        x,t,r,c,v = (a[active] for a in (x,t,r,c,v))
        def spatial(vec):
            return ((t-r)[:,None,None]*self._jvp(x,t,r,c,vec,0)).flatten(1)
        R = torch.stack([spatial(u.reshape(1,*x.shape[1:]).expand_as(x))
                         for u in self.dt_U.T],dim=-1)
        hv = spatial(v)
        return R, v.flatten(1)@self.dt_U, hv, v.flatten(1)-hv
