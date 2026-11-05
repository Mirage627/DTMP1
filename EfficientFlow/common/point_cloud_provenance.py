"""Record the installed observation implementation without inventing its semantics."""
import hashlib
import importlib
from pathlib import Path

from EfficientFlow.common.mp1_observation_contract import validate_point_cloud


def observation_provenance(env=None):
    names = {"robomimic.envs.env_robosuite", "robomimic.utils.obs_utils",
             "robosuite.environments.robot_env", "robosuite.environments.base"}
    if env is not None:
        names.add(type(env).__module__)
        if hasattr(env, "env"):
            names.add(type(env.env).__module__)
    modules = {}
    for name in sorted(names):
        try:
            module = importlib.import_module(name)
            filename = getattr(module, "__file__", None)
            path = Path(filename).resolve() if filename else None
            modules[name] = {"path":str(path) if path else None,
                             "sha256":hashlib.sha256(path.read_bytes()).hexdigest() if path and path.is_file() else None}
        except (ImportError, OSError) as error:
            modules[name] = {"unavailable":str(error)}
    return {"modules":modules, "coordinate_frame_verified":False,
            "color_scale_verified":False,
            "note":"Source fingerprints identify dependency versions; they do not certify XYZ frames, RGB scaling or point sampling."}


def check_generated_cloud(obs, *, required=False, expected_shape=(1024, 6)):
    if required or "point_cloud" in obs:
        cloud = validate_point_cloud(obs, "point_cloud", expected_shape)
        return {"shape":list(cloud.shape), "dtype":str(cloud.dtype),
                "min":cloud.min(0).tolist(), "max":cloud.max(0).tolist()}
    return None
