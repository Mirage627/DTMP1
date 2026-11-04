"""Shared, lightweight training dispatch; no model or dataset construction."""
import inspect
from pathlib import Path

import hydra
from omegaconf import OmegaConf


EFFICIENT_ADAFLOW_WORKSPACE = "EfficientFlow.workspace.train_equi_workspace.TrainEquiWorkspace"
EFFICIENT_ADAFLOW_POLICY = "EfficientFlow.policy.EfficientAdaFlow.EfficientAdaFlowPolicy"
MAX_STEPS = {
    "stack": 400, "stack_d0": 400, "stack_d1": 400,
    "stack_three_d1": 400, "square_d2": 400, "threading_d2": 400,
    "coffee_d2": 400, "three_piece_assembly_d2": 500,
    "hammer_cleanup_d1": 500, "mug_cleanup_d1": 500,
    "kitchen_d1": 800, "nut_assembly_d0": 500, "pick_place_d0": 1000,
    "coffee_preparation_d1": 800, "tool_hang": 700, "can": 400,
    "lift": 400, "square": 400,
}


def get_ws_x_center(task_name):
    return -0.2 if task_name.startswith(("kitchen_", "hammer_cleanup_")) else 0.0


def register_resolvers():
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.register_new_resolver("get_max_steps", lambda name: MAX_STEPS[name], replace=True)
    OmegaConf.register_new_resolver("get_ws_x_center", get_ws_x_center, replace=True)
    OmegaConf.register_new_resolver("get_ws_y_center", lambda name: 0.0, replace=True)


def run_workspace(cfg, *, source_root, entry_file, expected_workspace=None,
                  expected_policy=None, verify_only=False):
    """Resolve the configured workspace instead of hard-coding an MP1 loop.

    verify_only checks configuration and required source files. It deliberately
    avoids importing robot dependencies or reading the demonstration dataset.
    """
    OmegaConf.resolve(cfg)
    root = Path(source_root).resolve()
    workspace_target = OmegaConf.select(cfg, "_target_")
    policy_target = OmegaConf.select(cfg, "policy._target_")
    dataset_target = OmegaConf.select(cfg, "task.dataset._target_")
    if not isinstance(workspace_target, str) or not workspace_target:
        raise ValueError("The configuration must define its training workspace in _target_")
    if policy_target == EFFICIENT_ADAFLOW_POLICY:
        expected_workspace = EFFICIENT_ADAFLOW_WORKSPACE
    if expected_workspace is not None and workspace_target != expected_workspace:
        raise ValueError(
            f"Wrong workspace: {workspace_target}. This entry requires {expected_workspace}. "
            "Synchronize the complete EfficientAdaFlow configuration; do not use the MP1 training loop.")
    if expected_policy is not None and policy_target != expected_policy:
        raise ValueError(f"Wrong policy: {policy_target}. This entry requires {expected_policy}.")

    if policy_target == EFFICIENT_ADAFLOW_POLICY:
        required = (
            "config/EfficientAdaFlow.yaml", "policy/EfficientAdaFlow.py",
            "workspace/train_equi_workspace.py", "common/adaptive_flow.py",
            "common/equivariant_checkpoint.py", "model/equi/equi_conditional_unet1d.py",
            "model/equi/invariant_variance_head.py", "dataset/base_dataset.py",
            "dataset/robomimic_replay_image_dataset.py",
            "dataset/robomimic_replay_image_sym_dataset.py",
        )
        missing = [name for name in required if not (root / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Incomplete EfficientAdaFlow source at {root}: {missing}. "
                "Synchronize the project contents, including root entry files, common/ and workspace/.")
    print(f"ENTRY_FILE={Path(entry_file).resolve()}", flush=True)
    print(f"SOURCE_ROOT={root}", flush=True)
    print(f"WORKSPACE_TARGET={workspace_target}", flush=True)
    print(f"POLICY_TARGET={policy_target}", flush=True)
    print(f"DATASET_TARGET={dataset_target}", flush=True)
    if verify_only:
        print("ENTRY_CHECK_OK (configuration and source files only; training was not started)", flush=True)
        return
    workspace_class = hydra.utils.get_class(workspace_target)
    workspace_file = inspect.getsourcefile(workspace_class)
    print(f"WORKSPACE_FILE={workspace_file}", flush=True)
    workspace_class(cfg).run()
