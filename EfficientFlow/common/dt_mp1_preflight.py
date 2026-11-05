"""Exercise the existing vector runner before training, without new render contexts."""
import json
from pathlib import Path
import numpy as np
import torch
from EfficientFlow.common.mp1_observation_contract import validate_point_cloud
from EfficientFlow.common.point_cloud_provenance import observation_provenance


def rollout_preflight(runner, model, output, point_cloud_shape):
    """Schema checks cannot establish camera-frame or RGB-scale alignment."""
    was_training = model.training
    try:
        obs = runner.env.reset()
        expected = tuple(point_cloud_shape)
        clouds = np.asarray(obs.get('point_cloud', []))
        if clouds.ndim != 4 or clouds.shape[1] < model.n_obs_steps:
            raise ValueError(f'Expected [B,To,N,6] live point clouds; got {clouds.shape}')
        for cloud in clouds.reshape(-1, *clouds.shape[-2:]):
            validate_point_cloud({'point_cloud': cloud}, 'point_cloud', expected)
        tensor_obs = {key: torch.as_tensor(value, device=model.device, dtype=model.dtype)
                      for key, value in obs.items()}
        model.eval()
        model.reset()
        with torch.no_grad():
            result = model.predict_action(tensor_obs, initial_noise=torch.zeros(
                len(clouds), model.horizon, model.action_dim, device=model.device, dtype=model.dtype))
        action = result['action'].cpu().numpy()
        if not np.isfinite(action).all():
            raise ValueError('Non-finite policy action during live preflight')
        if runner.abs_action:
            action = runner.undo_transform_action(action)
        after, _, _, _ = runner.env.step(action)
        after_clouds = np.asarray(after.get('point_cloud', []))
        if after_clouds.shape != clouds.shape:
            raise ValueError('Live point cloud shape changed after env.step')
        for cloud in after_clouds.reshape(-1, *after_clouds.shape[-2:]):
            validate_point_cloud({'point_cloud': cloud}, 'point_cloud', expected)
        raw = clouds.reshape(-1, clouds.shape[-1])
        stats = model.normalizer['point_cloud'].get_input_stats()
        report = {'status': 'reset_inference_step_passed', 'point_cloud_shape': list(clouds.shape),
                  'observation_keys': sorted(obs),
                  'dtype': str(clouds.dtype), 'live_min': raw.min(0).tolist(),
                  'live_max': raw.max(0).tolist(),
                  'offline_train_min': stats['min'].cpu().tolist(),
                  'offline_train_max': stats['max'].cpu().tolist(),
                  'coordinate_frame_verified': False, 'color_scale_verified': False,
                  'runtime_observation_provenance': observation_provenance(),
                  'note': 'Schema/finite checks only. Verify XYZ frames, RGB scale and sampling against the original generator.'}
        Path(output, 'dt_rollout_preflight.json').write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding='utf8')
        return report
    except Exception as error:
        raise RuntimeError('DT-MP1 live rollout preflight failed before training. '
                           'Check the installed MimicGen tasks, XYZRGB generator and dataset env_args. '
                           f'Underlying error: {error}') from error
    finally:
        model.train(was_training)
        model.reset()
