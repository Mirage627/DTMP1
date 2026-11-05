# DT-MP1 审查修正与验证（2026-11-05）

本次接续原会话未完成的修改，对照 `DTMP1_code_audit_v3.md` 核对当前源码。
主公式、非对称 K 的转置、最小二乘展开、时间 JVP 和 Δ 缩放未发现确定性错误，不修改这些数学定义。
这是针对审查报告的工程修订，不代表已完成 v3 全部研究实验或证明性能提升。

## 修正结果

| 审查项 | 当前处理 |
| --- | --- |
| 仿真点云缺口 | wrapper 检查缺字段、shape、数值；训练前运行真实 runner reset → inference → step，失败中止。新增 `dt.preflight_only=true` 可只预检。 |
| 点云生成来源 | 用户提供的 `dataset_states_to_obs.py` 由 `EnvUtils.create_env_for_data_processing` 创建环境，实际 XYZRGB 由安装的环境依赖产生。新增依赖路径/SHA-256、相机配置记录和 `--require-point-cloud`；不猜测坐标系和采样方式。 |
| 训练数据数 | `dt_data_split.json` 记录实际训练/held-out 轨迹 ID、数量、窗口和批次数。1000 demos、0.16 held-out 对应 **840 条网络训练轨迹 + 160 条校准/验证轨迹**。 |
| 严格基线 | 使用同一 `DT_MP1` 配置、`dt.family=baseline`。零校正保持原 MP1 全批 primal/JVP 计算路径，避免近零梯度差异被 AdamW 放大。 |
| 校准成本 | 删除固定 1000 窗口摊销假设，用实际剩余训练批次与实测 baseline 梯度耗时分配成本，计入 report 阶段耗时。该指标仍只覆盖梯度计算，不等于真实总 GPU 小时。 |
| K 的寿命 | 保存 `dt_early/middle/late.ckpt`；独立脚本验证早 K → 中/晚、中 K → 晚，报告方差、无偏平方漂移估计、计算预算风险和诊断回退比例。 |
| 结构验证 | diagonal 必须严格对角；sub_scalar 必须等比例单位阵；非有限值、错误维度和过大增益拒绝，失败不改变当前 K。 |
| 日志 | 记录非对角时间样本比例、校正范数、K 范数、回退、裁剪前后范数、裁剪比例/缩放、epoch 耗时、CUDA 峰值显存和 rollout 耗时。全局范数裁剪仅按一个正标量缩放，理论方向不变。 |
| 图像开销 | 新增 DT 专用任务配置，训练 dataset/policy 只读取点云和状态；runner 保留两路相机用于可视化。 |
| rollout 调度 | 默认最少间隔 5 epochs，1000 demos 从每 epoch rollout 改为每 5 epochs。所有实验分支应共享这一配置。 |
| alpha 含义 | 保留旧配置键 `dt.alpha`，明确其为 bootstrap 显著性水平，不是方向修正强度。 |

根目录的 `VALIDATION.md` 记录初版发布时的验证。本次修正的验证结果和文件清单位于本文所在的 `docs` 目录。

## 点云检查

原数据生成命令增加一个严格检查开关即可；输出应使用新文件名，脚本现在拒绝覆盖已存在的数据集：

```bash
python EfficientFlow/scripts/dataset_states_to_obs.py \
  --input data/robomimic/datasets/kitchen_d1/kitchen_d1.hdf5 \
  --output data/robomimic/datasets/kitchen_d1/kitchen_d1_voxel_checked.hdf5 \
  --num_workers=24 --require-point-cloud
```

数据生成时在 HDF5 `data` 的 `observation_provenance` 属性中记录依赖来源与相机配置。该脚本只生成观测，**不会把相对动作自动转换成绝对动作**；DT 训练仍需已有绝对动作数据。已有正确数据无需重新生成。

在服务器 DTMP1 仓库根目录，只运行真实环境预检：

```bash
python -u train.py --config-name=DT_MP1 \
  task_name=kitchen_d1 n_demo=1000 \
  dt.preflight_only=true logging.mode=disabled \
  task.env_runner.n_envs=1 \
  hydra.run.dir=runs/DTMP1_live_preflight_20261105
```

输出 `dt_rollout_preflight.json` 包含观测 key、shape、dtype、实时/离线六通道范围与运行时依赖来源。
`coordinate_frame_verified=false` 和 `color_scale_verified=false` 是有意保留的事实：shape 和有限值一致无法证明坐标系、RGB 尺度、裁剪/FPS 一致。
训练初期的预检执行的是未训练策略，只用于接口检查；不能据此评价任务成功率。

## 跨 checkpoint 诊断

训练仍然是 warmup 后校准一次并冻结 K。独立诊断不会改变训练流程或原 checkpoint：

```bash
RUN=runs/你的DTMP1训练目录
python EfficientFlow/scripts/diagnose_dt_mp1_lifetime.py \
  --early "$RUN/checkpoints/dt_early.ckpt" \
  --middle "$RUN/checkpoints/dt_middle.ckpt" \
  --late "$RUN/checkpoints/dt_late.ckpt" \
  --output-dir runs/DTMP1_lifetime_20261105 \
  --device cuda:0 --pairs 16
```

诊断在中期模型上单独重新校准一次，成本记录于 `middle_refit/dt_calibration.json`；配置为 baseline 时跳过拟合。跨阶段评估把已固定 K 的校准成本视为沉没成本。
`dt_lifetime.json` 的回退是诊断判定，不会回写模型。报告中的平方漂移估计允许出现负值，因为无偏估计有采样噪声。
同一 report 轨迹跨阶段复用，因此这些是探索性配对诊断；不能据此反复调参后再把该集合当独立最终测试集。两个 epoch 的合成测试中 middle/late 会重合，真实实验应使用足够长训练获得三个不同阶段。

## 验证与未解决项

本地使用 Windows、Python 3.9、PyTorch CPU 和小宽度模型；真实任务形状测试含 1024×6 点云。
测试命令（发布目录布局）：

```bash
python -B -m unittest discover -s EfficientFlow/tests -p test_dt_mp1.py -v
python -B train.py --verify --config-name=DT_MP1 task_name=coffee_preparation_d1 n_demo=1000
```

覆盖零 K 全参数梯度与一次 AdamW 更新（两个 batch size）、非对称 K/空间 JVP 标签恒等式、family 原子校验、训练归一化隔离、轨迹划分、HDF5 shape、预检失败、点云生成检查、校准成本、训练/恢复/共同 warmup 分叉和跨阶段诊断。实际运行结果见 [验证记录](VALIDATION_DT_MP1.txt)。

尚未完成的实证项：真实仿真 reset/infer/step、两种设备一致性、完整模型 CUDA 显存/耗时、真实早中晚 K 寿命、多训练 seed/轨迹簇不确定性，以及含校准开销的等 GPU 小时 rollout。不能把 16 对批次 bootstrap 解读为论文统计功效或成功率保证。
本地未安装完整 robotics 环境；本次只读 SSH 连接被服务器以 `Permission denied (publickey,password)` 拒绝，未改服务器代码或正在运行的训练。

新配置移除了未使用的图像输入并改变 rollout 频率，旧 checkpoint 的严格配置签名可能不匹配。已有训练继续使用原配置；新对照实验从相同的新配置开始。不要在一组对照中混用旧/新配置。

## 文件同步

在 Windows PowerShell 中进入包含本次修正的 DTMP1 本地仓库根目录。填写服务器 SSH 地址和远端仓库绝对路径，命令按清单逐个上传文件并保留相对路径。

```powershell
$localRepo = (Get-Location).Path
$server = Read-Host '服务器 SSH 地址 (user@host)'
$remoteRepo = (Read-Host '服务器 DTMP1 仓库绝对路径').TrimEnd('/')
if ($remoteRepo -notmatch '^/[A-Za-z0-9_./-]+$') { throw '请输入不含空格的 Linux 绝对路径' }
$manifestPath = 'EfficientFlow/docs/DTMP1_audit_manifest_20261105.json'
$manifest = Get-Content -LiteralPath (Join-Path $localRepo $manifestPath) -Raw | ConvertFrom-Json
$files = @($manifest.sha256.PSObject.Properties.Name) + @($manifestPath, 'README.md', 'SOURCE_MANIFEST.json')
ssh $server "test -f '$remoteRepo/EfficientFlow/policy/dt_mp1.py' && mkdir -p '$remoteRepo/EfficientFlow/docs'"
if ($LASTEXITCODE -ne 0) { throw 'SSH 登录失败或远端路径不是 DTMP1 仓库' }
foreach ($file in $files) {
    scp (Join-Path $localRepo $file) "${server}:$remoteRepo/$file"
    if ($LASTEXITCODE -ne 0) { throw "上传失败：$file" }
}
```

在服务器的 DTMP1 仓库根目录运行回归测试。

```bash
python -B -m unittest discover -s EfficientFlow/tests -p test_dt_mp1.py -v
```

审查修正的 SHA-256 见 [文件清单](DTMP1_audit_manifest_20261105.json)。上传会更新清单中的源码，请在启动新实验前完成。
