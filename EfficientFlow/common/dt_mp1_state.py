"""Frozen, portable DT correction state in normalized action coordinates."""
from dataclasses import dataclass, field
import hashlib

import torch


FAMILIES = ("baseline", "global_scalar", "sub_scalar", "diagonal", "full")


def tensor_fingerprint(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(str((name, str(value.dtype), tuple(value.shape))).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def normalizer_fingerprint(model):
    return tensor_fingerprint(model.normalizer.state_dict())


def validate_basis(basis, dimension):
    u = torch.as_tensor(basis).detach()
    if u.ndim != 2 or u.shape[0] != dimension or not 1 <= u.shape[1] <= dimension:
        raise ValueError("DT basis must have shape [action dimension, rank], 1 <= rank <= dimension")
    if not u.is_floating_point() or not torch.isfinite(u).all():
        raise ValueError("DT basis must be finite floating point")
    gram = u.double().T @ u.double()
    if (gram-torch.eye(u.shape[1], device=u.device)).abs().max() > 2e-5:
        raise ValueError("DT basis is not orthonormal")
    return u


def validate_correction(basis, family, gain, dimension):
    u = validate_basis(basis, dimension)
    k = u.new_zeros((u.shape[1], u.shape[1]))
    g = u.new_zeros(())
    if family not in FAMILIES:
        raise ValueError(f"Unknown DT family: {family}")
    if family == "baseline":
        if gain is not None and torch.as_tensor(gain).ne(0).any():
            raise ValueError("Baseline cannot carry a nonzero gain")
        return u, k, g
    value = torch.as_tensor(gain, device=u.device, dtype=u.dtype).detach()
    if not torch.isfinite(value).all():
        raise ValueError("Non-finite DT gain")
    if family == "global_scalar" and value.ndim == 0:
        g.copy_(value)
        size = g.abs()
    elif family in FAMILIES[2:] and value.shape == k.shape:
        if family in ("diagonal", "sub_scalar") and not torch.equal(value, torch.diag(value.diagonal())):
            raise ValueError(f"{family} requires an exactly diagonal gain")
        if family == "sub_scalar" and not torch.equal(value, torch.eye(len(k),device=k.device)*value[0,0]):
            raise ValueError("sub_scalar requires a scalar multiple of identity")
        k.copy_(value)
        size = torch.linalg.matrix_norm(k, ord=2)
    else:
        raise ValueError(f"Wrong DT gain shape for {family}: {value.shape}")
    if not torch.isfinite(size) or size > 2:
        raise ValueError("Invalid DT gain: spectral bound is 2")
    return u, k, g


@dataclass(frozen=True)
class Correction:
    basis: torch.Tensor
    family: str
    gain: object = None
    metadata: dict = field(default_factory=dict)

    @classmethod
    def capture(cls, model, metadata=None):
        family = FAMILIES[int(model.dt_family)]
        gain = model.dt_global if family == "global_scalar" else model.dt_K
        return cls(model.dt_U.detach().cpu().clone(), family,
                   None if family == "baseline" else gain.detach().cpu().clone(),
                   {"normalizer_sha256":normalizer_fingerprint(model), **(metadata or {})})

    def as_dict(self):
        return {"basis":self.basis.tolist(), "family":self.family,
                "gain":None if self.gain is None else self.gain.tolist(), "metadata":self.metadata}


def apply_correction(correction, *models):
    """Validate every target before changing any; keep main and EMA buffers identical."""
    prepared = [(m, m.prepare_correction(correction)) for m in models if m is not None]
    for model, values in prepared:
        model._install_correction(*values)
