# DTMP1

DT-MP1 v2.1：在 MP1 上对训练时的空间 JVP 方向进行校准。保留 `EfficientFlow.*` 导入路径，提供独立策略、训练器、六通道点云配置、数据读取、仿真资源与 CPU 测试。

2026-11-05 审查修正加入点云预检、校准和裁剪日志、跨 checkpoint 的 K 诊断。逐文件同步命令见 [审查修正说明](EfficientFlow/docs/DTMP1_audit_response_20261105_zh.md)。

## 算法与文件

`policy/dt_mp1.py` 继承原 MP1。训练先进行公共 warmup，再利用与训练轨迹分离的 fit / tune / accept / report 数据分区，拟合并筛选固定方向修正；校准完成后冻结修正矩阵。默认候选是 diagonal，同时保留标量和零修正对照。未通过筛选时允许退回标量或原 MP1 方向。

设原空间方向为 `d_w`，子空间基为 `U`，校准矩阵为 `K`，修正方向为 `d_w + U K Uᵀ (d_w - boundary)`。修正只作用于空间 JVP，时间方向和原 MP1 leading target 保持原定义。实现不需要另一个网络预测速度。它改变训练估计量，可能引入偏差；经验梯度风险筛选不能证明任务成功率提高，也不代表彻底消除 JVP 随机性。

以下路径相对于 `EfficientFlow/`：

| 内容 | 文件 |
| --- | --- |
| DT 策略与方向修正 | `policy/dt_mp1.py` |
| 训练集归一化、校准、筛选 | `common/dt_mp1_calibration.py` |
| 公共 warmup、恢复、checkpoint、进度条 | `workspace/train_dt_mp1_workspace.py` |
| 六通道 XYZRGB 训练配置 | `config/DT_MP1.yaml` |
| 回归测试 | `tests/test_dt_mp1.py` |
| 原 MP1 参考 | `policy/mp1.py`、`config/MP1.yaml` |
| 数据、rollout 与环境 | `dataset/`、`env/`、`env_runner/`、`gym_util/` |
| MimicGen 及运行所需网格与纹理 | `mimicgen_environments/` |

本仓库是独立源码快照，不包含数据集、checkpoint、实验输出、凭据或原仓库 Git 历史。`SOURCE_MANIFEST.json` 记录来源文件和校验值。

## 安装

面向 Linux + NVIDIA CUDA 训练。在已能运行 EfficientFlow 的环境中，从本仓库根目录执行：

```bash
conda activate EfficientFlow
python -m pip install -r requirements-dtmp1.txt
python -m pip install --no-deps -e .
python -m pip install --no-deps -e EfficientFlow/mimicgen_environments
```

新环境可使用以下起点；完整环境尚未在全新 GPU 机器重建验证：

```bash
conda env create -f conda_environment.yaml
conda activate EfficientFlow
python -m pip install --no-deps -e .
python -m pip install --no-deps -e EfficientFlow/mimicgen_environments
```

环境文件基于原项目 Python 3.9 / PyTorch 2.1 / CUDA 11.8 / PyTorch3D 0.7.5，采用本地 CPU 验证用的 Diffusers 0.27.2 / Hugging Face Hub 0.25.2，并补充 SwanLab。原环境文件保存在 `environment/conda_environment.upstream.yaml`。仿真依赖中如有旧的 `import mimicgen_envs`，需要与本仓库安装的 `mimicgen` 包名匹配。无显示器服务器可设置：

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

## 数据与训练

从仓库根目录运行，先准备绝对动作、1024 点 XYZRGB 观测的 HDF5 文件：

```text
data/robomimic/datasets/coffee_preparation_d1/coffee_preparation_d1_voxel_abs.hdf5
```

也可通过 `dataset_path=/absolute/path/dataset.hdf5` 指定已有数据。转换脚本保存在 `EfficientFlow/scripts/robomimic_dataset_conversion.py` 和 `robomimic_dataset_obs_conversion.py`，原数据准备说明见 [上游 README](EfficientFlow/README.md)。训练前会检查请求的 demo 数量、观测 shape 和动作编码，不会自动将 1000 demos 缩减为较少数据。

```bash
python -u train.py --verify --config-name=DT_MP1 \
  task_name=coffee_preparation_d1 n_demo=1000

swanlab login
python -u train.py --config-name=DT_MP1 \
  task_name=coffee_preparation_d1 n_demo=1000 \
  logging.project=MP1_2 \
  logging.name=coffee_preparation_d1_1000_DT_MP1_s10 \
  training.seed=10 \
  hydra.run.dir=runs/DT_MP1_coffee_preparation_d1_1000_s10
```

该命令使用六通道点云、4 步预测、2 步观测，执行预测序列的后 3 步。1000 demos 中有 840 条用于训练，160 条留给校准和验证。共训练 50 epochs，前 5 epochs 为 warmup，默认每 5 epochs 运行 rollout。`dt_data_split.json` 记录轨迹划分，`dt_calibration.json` 保存筛选结果，`dt_metrics.jsonl` 保存训练指标。断点恢复时使用同一 `hydra.run.dir` 和相同训练配置。

不使用在线日志可加 `logging.mode=offline`；只检查训练流程可加 `training.enable_rollouts=false`。`--verify` 只检查配置解析与入口，不读取真实数据或启动仿真。

公平的零修正对照仍使用 `DT_MP1` 配置，增加 `dt.family=baseline` 并使用独立输出目录。`config/MP1.yaml` 是历史参考，不是这个对照的替代配置。

## 测试与评估

```bash
python -B -m unittest discover -s EfficientFlow/tests -p test_dt_mp1.py -v

python EfficientFlow/scripts/eval_mp1_checkpoint.py \
  --checkpoint runs/DT_MP1_coffee_preparation_d1_1000_s10/checkpoints/latest.ckpt \
  --output-dir runs/DT_MP1_eval --mode offline --n-envs 2
```

当前验证覆盖零修正的参数梯度和 AdamW 更新、六通道接口、动作窗口、归一化隔离、校准筛选、保存恢复和跨阶段诊断。详见 [本次验证记录](EfficientFlow/docs/VALIDATION_DT_MP1.txt) 和 [初版验证记录](VALIDATION.md)。

## 来源与许可证

基于 [EfficientFlow 源码](https://github.com/Mirage627/PROJECT_EF) 和 MP1 扩展，保留原作者说明与引用于 [上游 README](EfficientFlow/README.md)。主项目保留 [MIT License](LICENSE)；MimicGen 遵循其 [NVIDIA Source Code License](EfficientFlow/mimicgen_environments/LICENSE)。第三方文件中的版权与许可声明予以保留。
