"""Inference utilities for interval-average MeanFlow velocity fields.

Schedule construction intentionally has no torch dependency. Sampling imports
torch lazily so experiment definitions can be inspected without a GPU runtime.
"""

import math
from numbers import Integral


def _validate_num_steps(num_inference_steps):
    if (isinstance(num_inference_steps, bool)
            or not isinstance(num_inference_steps, Integral)
            or num_inference_steps < 1):
        raise ValueError("num_inference_steps must be a positive integer")
    return int(num_inference_steps)


def validate_time_grid(time_grid):
    """Return a finite, strictly decreasing 1-to-0 grid as a float tuple."""
    if isinstance(time_grid, (str, bytes)):
        raise ValueError("time_grid must be a sequence of numbers, not a string")
    try:
        values = tuple(time_grid)
    except TypeError as exc:
        raise ValueError("time_grid must be a sequence of numbers") from exc
    if len(values) < 2:
        raise ValueError("time_grid must contain at least the endpoints 1 and 0")
    if any(isinstance(value, bool) for value in values):
        raise ValueError("time_grid entries must be numbers, not booleans")
    try:
        grid = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError("time_grid must be a one-dimensional numeric sequence") from exc
    if not all(math.isfinite(value) for value in grid):
        raise ValueError("time_grid entries must be finite")
    if grid[0] != 1.0 or grid[-1] != 0.0:
        raise ValueError("time_grid must start at 1 and end at 0")
    if not all(0.0 <= value <= 1.0 for value in grid):
        raise ValueError("time_grid entries must lie in [0, 1]")
    if not all(t > r for t, r in zip(grid[:-1], grid[1:])):
        raise ValueError("time_grid must be strictly decreasing")
    return grid


def make_time_grid(num_inference_steps=1, schedule="uniform", power=2.0):
    """Construct uniform, noise_first, or data_first inference intervals.

    With power > 1, noise_first allocates shorter intervals near noise (t=1),
    whereas data_first allocates shorter intervals near data (t=0).
    """
    steps = _validate_num_steps(num_inference_steps)
    try:
        power = float(power)
    except (TypeError, ValueError) as exc:
        raise ValueError("power must be finite and positive") from exc
    if not math.isfinite(power) or power <= 0:
        raise ValueError("power must be finite and positive")
    aliases = {
        "uniform": "uniform",
        "noise_first": "noise_first",
        "noise_dense": "noise_first",
        "front_loaded": "noise_first",
        "data_first": "data_first",
        "data_dense": "data_first",
        "back_loaded": "data_first",
    }
    try:
        family = aliases[schedule]
    except (KeyError, TypeError) as exc:
        raise ValueError("schedule must be uniform, noise_first, or data_first") from exc
    if family == "noise_first":
        grid = [1.0 - (index / steps) ** power for index in range(steps + 1)]
    elif family == "data_first":
        grid = [(1.0 - index / steps) ** power for index in range(steps + 1)]
    else:
        grid = [1.0 - index / steps for index in range(steps + 1)]
    grid[0], grid[-1] = 1.0, 0.0
    return validate_time_grid(grid)


def resolve_time_grid(time_grid=None, num_inference_steps=None):
    """Resolve an explicit grid or a uniform step count; default to one step."""
    if time_grid is None:
        return make_time_grid(1 if num_inference_steps is None else num_inference_steps)
    grid = validate_time_grid(time_grid)
    if num_inference_steps is not None:
        steps = _validate_num_steps(num_inference_steps)
        if steps != len(grid) - 1:
            raise ValueError("num_inference_steps does not match time_grid intervals")
    return grid


def sample_meanflow(velocity_fn, condition_data, condition_mask=None, *,
                    time_grid=None, num_inference_steps=None,
                    initial_noise=None, generator=None):
    """Sample using ``z_r = z_t - (t-r) * velocity_fn(z_t, t, r)``.

    velocity_fn receives batched time tensors and must return a tensor with the
    same shape as condition_data. Each interval invokes it again with the new
    state and both endpoints; a whole-interval average cannot be reused for a
    shorter interval. Known/inpainted values are imposed before every model
    call and after every update. The supplied initial_noise is never mutated.
    It takes precedence over generator and must match shape, dtype, and device.

    This helper performs exactly len(time_grid)-1 velocity evaluations and
    returns only the sampled tensor. It does not change any module's mode.
    """
    import torch

    grid = resolve_time_grid(time_grid, num_inference_steps)
    if not isinstance(condition_data, torch.Tensor) or condition_data.ndim < 1:
        raise ValueError("condition_data must be a batched torch tensor")
    if not condition_data.is_floating_point():
        raise ValueError("condition_data must have a floating point dtype")
    if condition_mask is not None:
        if (not isinstance(condition_mask, torch.Tensor)
                or condition_mask.shape != condition_data.shape
                or condition_mask.dtype != torch.bool
                or condition_mask.device != condition_data.device):
            raise ValueError("condition_mask must be a bool tensor matching condition_data")
    if initial_noise is not None:
        if (not isinstance(initial_noise, torch.Tensor)
                or initial_noise.shape != condition_data.shape
                or initial_noise.dtype != condition_data.dtype
                or initial_noise.device != condition_data.device):
            raise ValueError("initial_noise must match condition_data shape, dtype, and device")

    with torch.no_grad():
        if initial_noise is None:
            z = torch.randn(condition_data.shape, dtype=condition_data.dtype,
                            device=condition_data.device, generator=generator)
        else:
            z = initial_noise.detach().clone()
        for t_value, r_value in zip(grid[:-1], grid[1:]):
            if condition_mask is not None:
                z = torch.where(condition_mask, condition_data, z)
            # Match the original torch.ones/zeros time-conditioning dtype,
            # independently of the action tensor's precision.
            t = torch.full((z.shape[0],), t_value, device=z.device,
                           dtype=torch.get_default_dtype())
            r = torch.full((z.shape[0],), r_value, device=z.device,
                           dtype=torch.get_default_dtype())
            velocity = velocity_fn(z, t, r)
            if not isinstance(velocity, torch.Tensor) or velocity.shape != z.shape:
                raise ValueError("velocity_fn must return a tensor matching the sample shape")
            z = z - (t_value - r_value) * velocity
            if condition_mask is not None:
                z = torch.where(condition_mask, condition_data, z)
    return z
