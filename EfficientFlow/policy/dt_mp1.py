"""MP1 spatial-tangent calibration with explicit gain-family contracts."""
import math
import torch
from EfficientFlow.policy.mp1 import Meanpolicy, adaptive_l2_loss
from EfficientFlow.common.pytorch_util import dict_apply
from EfficientFlow.common.dt_mp1_state import (Correction, FAMILIES, normalizer_fingerprint,
    tensor_fingerprint, validate_correction)


def action_basis(horizon, action_dim, mode="dct_all", rank=None):
    """Static ablations: equal time-frequency coverage for every action channel.

    The historical first-four-channel basis is explicit opt-in only. A balanced
    DCT rank must contain whole frequency blocks, never a prefix of channels.
    """
    if mode not in ("dct_all", "legacy_first4"):
        raise ValueError(f"Unknown static DT basis: {mode}")
    channels = min(4, action_dim) if mode == "legacy_first4" else action_dim
    frequencies = min(2, horizon) if rank is None else rank // channels
    if rank is not None and (rank % channels or not 1 <= frequencies <= horizon):
        raise ValueError("Static DT rank must contain complete action-channel blocks")
    basis = []
    for ch in range(channels):
        for freq in range(frequencies):
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
        self.register_buffer("dt_family", torch.zeros((), dtype=torch.int64))

    @torch.no_grad()
    def set_gain(self, family, gain=None):
        self.set_correction(Correction(self.dt_U, family, gain))

    @torch.no_grad()
    def set_basis(self, basis):
        if self.dt_K.ne(0).any() or self.dt_global.ne(0):
            raise ValueError("Reset gain before changing U; old K cannot be reused with a new basis")
        self.set_correction(Correction(basis, "baseline"))

    def prepare_correction(self, correction):
        expected = correction.metadata.get("normalizer_sha256")
        if expected is not None and expected != normalizer_fingerprint(self):
            raise ValueError("DT correction normalizer fingerprint mismatch")
        u = correction.basis.to(device=self.dt_U.device, dtype=self.dt_U.dtype)
        u,k,g = validate_correction(u, correction.family, correction.gain, self.horizon*self.action_dim)
        return u.clone(), k.clone(), g.clone(), FAMILIES.index(correction.family)

    @torch.no_grad()
    def _install_correction(self, u, k, g, family):
        self.dt_U, self.dt_K, self.dt_global = u, k, g
        self.dt_family.fill_(family)

    def set_correction(self, correction):
        self._install_correction(*self.prepare_correction(correction))

    def set_normalizer(self, normalizer):
        if (self.dt_K.ne(0).any() or self.dt_global.ne(0)) and (
                tensor_fingerprint(normalizer.state_dict()) != normalizer_fingerprint(self)):
            raise ValueError("Reset DT gain before changing the normalizer coordinates")
        super().set_normalizer(normalizer)

    @staticmethod
    def _integrity(state, prefix):
        fields = {k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix) and
                  (k[len(prefix):] in ("dt_U", "dt_K", "dt_global", "dt_family") or
                   k[len(prefix):].startswith("normalizer."))}
        return torch.tensor(list(bytes.fromhex(tensor_fingerprint(fields))), dtype=torch.uint8)

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        state = super().state_dict(*args, destination=destination, prefix=prefix, keep_vars=keep_vars)
        state[prefix+"dt_integrity"] = self._integrity(state, prefix)
        return state

    def _load_from_state_dict(self, state, prefix, metadata, strict, missing, unexpected, errors):
        # Verify before mutation, and resize registered buffers for a calibrated rank.
        u,k,g = (state[prefix+n] for n in ("dt_U", "dt_K", "dt_global"))
        digest = state.pop(prefix+"dt_integrity", None)
        if digest is not None:
            if not torch.equal(digest.cpu(), self._integrity(state, prefix)):
                raise ValueError("DT checkpoint U/K/normalizer fingerprint mismatch")
            family = FAMILIES[int(state[prefix+"dt_family"])]
        else:
            # Explicit compatibility with pre-v4 checkpoints only.
            legacy = action_basis(self.horizon, self.action_dim, "legacy_first4")
            if prefix+"dt_family" in state or u.shape != legacy.shape or not torch.equal(u.cpu(),legacy):
                raise ValueError("Missing DT checkpoint integrity fingerprint")
            family = "global_scalar" if g.ne(0) else ("full" if k.ne(0).any() else "baseline")
            state[prefix+"dt_family"] = torch.tensor(FAMILIES.index(family))
        gain = g if family == "global_scalar" else (None if family == "baseline" else k)
        _,expected_k,expected_g = validate_correction(u, family, gain, self.horizon*self.action_dim)
        if not torch.equal(k,expected_k) or not torch.equal(g,expected_g):
            raise ValueError("Inconsistent DT family and gain buffers")
        self.dt_U = torch.empty_like(u, device=self.dt_U.device)
        self.dt_K = torch.empty_like(k, device=self.dt_K.device)
        super()._load_from_state_dict(state,prefix,metadata,strict,missing,unexpected,errors)

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
    def basis_rows(self, batch):
        """Return detached v, ell, H.T ell; H = (t-r) J_x, with frozen c,t,r."""
        x,t,r,c,dw,v = self._draw(batch)
        delta = t-r
        active = delta > 0
        if not active.any():
            return None, delta
        x,t,r,c,v = (a[active].detach() for a in (x,t,r,c,v))
        hv = (t-r)[:,None,None]*self._jvp(x,t,r,c,v,0)
        ell = (v-hv).detach()
        with torch.enable_grad():
            xx = x.detach().requires_grad_(True)
            prediction = self.model(xx,t,r=r,global_cond=c,training=False)
            hte = torch.autograd.grad(prediction, xx,
                grad_outputs=((t-r)[:,None,None]*ell), create_graph=False)[0]
        return (v.flatten(1), ell.flatten(1), hte.detach().flatten(1)), delta

    @torch.no_grad()
    def correction_ratios(self, batch):
        _,t,r,_,dw,v = self._draw(batch)
        active = t > r
        change = self.tangent(dw,v,active)-dw
        return (change.flatten(1).norm(dim=1)/(dw.flatten(1).norm(dim=1)+1e-8))[active], t-r

    @torch.no_grad()
    def calibration_rows(self, batch, *, need_basis=True):
        x,t,r,c,_,v = self._draw(batch)
        active = t > r
        if not active.any():
            return None
        x,t,r,c,v = (a[active] for a in (x,t,r,c,v))
        def spatial(vec):
            return ((t-r)[:,None,None]*self._jvp(x,t,r,c,vec,0)).flatten(1)
        R = (torch.stack([spatial(u.reshape(1,*x.shape[1:]).expand_as(x))
                         for u in self.dt_U.T],dim=-1) if need_basis else x.new_empty(len(x),x[0].numel(),0))
        hv = spatial(v)
        return R, v.flatten(1)@self.dt_U, hv, v.flatten(1)-hv
