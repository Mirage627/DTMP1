# DT-MP1 source snapshot validation

Date: 2026-11-04. Validation was performed from this exported repository, retaining the `EfficientFlow` source namespace.

## Executed checks

- `python -B -u -m unittest discover -s EfficientFlow/tests -p test_dt_mp1.py -v`: **7 tests passed**.
- `python -B -u train.py --verify --config-name=DT_MP1 task_name=coffee_preparation_d1 n_demo=1000`: **passed**. Resolved policy, workspace and dataset targets use the exported source files.
- Python source syntax compilation, Hydra target file availability, source-manifest integrity and publication file checks are performed before the initial commit.

The tests cover zero-gain parameter-gradient equivalence with MP1, an explicit corrected-target comparison, non-symmetric and diagonal gain handling, real coffee observation/action shapes, train-only normalization, disjoint episode partitions, fit/gate controls, HDF5 metadata rejection, calibration, checkpoint resume and a common-warmup branch.

The local test environment is Windows, Python 3.9.15 and PyTorch 2.1 CPU. Existing task-local dependencies provided Diffusers 0.27.2 and Hugging Face Hub 0.25.2 without changing the user's running experiment environment. The tests use small model widths and synthetic replay data. The coffee shape test uses 1024-point, six-channel observations; it does not train a full-size model.

## Limits

No complete real-dataset training, CUDA memory/performance measurement, simulator rollout or task-success comparison was executed for this publication. The full Conda environment was not rebuilt on a clean Linux machine. HDF5 preflight tests check schema and shape handling; they do not certify a real dataset's action semantics or temporal alignment. Algorithmic effectiveness still requires controlled experiments against the common-warmup, zero-correction control.

The source repository contains other active work. This snapshot reads the working-tree source files and exports only the required dependency closure and MimicGen runtime assets. Its independent Git history does not modify the source repository's index or remotes. `SOURCE_MANIFEST.json` records the base revision, hashes of source bytes and hashes of normalized published files; the base revision alone does not identify the uncommitted algorithm additions.
