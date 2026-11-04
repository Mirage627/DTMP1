"""Read small HDF5 slices to inspect MP1 camera/action/color metadata.

Does not load a checkpoint, start MuJoCo, or modify the dataset.
"""
import argparse
import json


def inspect_dataset(path, demo_index=0):
    import h5py
    import numpy as np

    def summary(values):
        values = np.asarray(values)
        flat = values.reshape(-1, values.shape[-1])
        return {"shape": list(values.shape), "dtype": str(values.dtype),
                "finite": bool(np.isfinite(values).all()),
                "min": flat.min(0).tolist(), "max": flat.max(0).tolist(),
                "mean": flat.mean(0).tolist(), "std": flat.std(0).tolist()}

    with h5py.File(path, "r") as file:
        meta = json.loads(file["data"].attrs["env_args"])
        demo = file[f"data/demo_{demo_index}"]
        report = {"dataset": str(path), "env_args": meta, "demo": demo_index,
                  "has_model_file": "model_file" in demo.attrs,
                  "steps": len(demo["actions"]), "samples": {}}
        length = len(demo["actions"])
        indices = np.unique(np.linspace(0, length-1, min(length, 8), dtype=int))
        if not len(indices):
            raise ValueError("Empty demonstration")
        report["sampled_indices"] = indices.tolist()
        report["samples"]["actions"] = summary(demo["actions"][indices])
        for key in ("point_cloud", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"):
            if key in demo["obs"]:
                report["samples"][key] = summary(demo["obs"][key][indices])
        report["observation_shapes"] = {key: list(value.shape) for key, value in demo["obs"].items()}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--demo", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(inspect_dataset(args.dataset_path, args.demo), indent=2))


if __name__ == "__main__":
    main()
