# DT-MP1：U 设计复核与本次修改

复核对象：用户提供的 `DTMP1_ideal_algorithm_and_code_changes.md`，以及追加的《对我提出的自适应 U 算法重新评估》。本文区分可验证的数学结论、实现缺陷和仍需实验验证的研究建议。

## 判断

1. **旧 U 数学上成立，但不适合作为无说明的默认基。** Kitchen 的动作是每步 XYZ、6D rotation、gripper，共 10 维，4 步共 40 维。旧基仅选择前 4 个通道、每通道 2 个时间频率，遗漏其余旋转分量和夹爪。它是合法的受限消融，不能据此声称覆盖了主要噪声方向；8/40 也不能换算成保留或丢失的降噪收益比例。
2. **报告的代理梯度推导正确。** 对 `J(B)=E||ell-H B v||²+lambda||B||F²`，在零点有 `grad J(0)=-2 M`，其中 `M=E[(H.T ell)v.T]`。输入 VJP 中的余切必须是 `Delta*ell`，而 `ell=v-Delta*J_x v` 必须停止梯度。测试用解析线性模型和中心有限差分验证这一点。
3. **`sym(M)` 的绝对主特征向量不是一般意义的最优 U。** 它适合作为允许正负增益的对称一阶候选，但忽略曲率、代理误差，也忽略 M 的反对称部分。例如 `M=[[0,1],[-1,0]]` 时 `sym(M)=0`，完整非对称 B 仍有非零下降方向。报告已承认启发式性质，因此这一方案可以用于候选发现，不能直接等同于最优降噪器。
4. **代理回归、实际修正量检查和独立完整梯度验收是不同的环节。** 前两项不能替代后一项；冻结梯度风险也不是任务成功率证明。沿用有偏差风险的 semi-gradient 解释，不宣称无偏。
5. **补充报告的二阶收益公式正确，但有适用条件。** 对固定单位方向，记 `f=E[(u.T v)(H u).T ell]`、`c=E[(u.T v)^2||H u||²]`，无约束的 ridge 解为 `kappa=f/(c+eta)`，正则化目标下降为 `f²/(c+eta)`。这需要 `c+eta>0`，没有加入谱范数/尾部拒绝约束，且不能把多方向耦合回归的收益简单相加。用 `H=diag(2,0.4)`、`Cov(v)=diag(0.1,0.8)`、`eta=0.01` 可验证：第一方向的绝对一阶分数较大，第二方向的实际 ridge 收益更大，而且两者最优增益都在上限 2 内。本次以实际 K 回归和独立完整梯度风险比较有限候选 U，不把一阶谱排序当成最终选择依据。
6. **“不同时间状态抵消”需要区别模型假设。** 若完整 M（不是仅 sym(M)）为零，则固定全局 B 的目标是 `常数 + E||H B v||² + eta||B||F²`，零修正就是全局最优之一；仅重选 U 不能改善这一目标。各桶使用不同 K 可以消除抵消，但扩大了模型类，不能据此断言原全局 U 错误。本次增加分桶矩阵强度诊断，不自动学习分桶 K。
7. **周期重校准和两侧不同子空间是另一个研究阶段。** 默认仍为 warmup 后一次联合校准，没有无条件周期刷新，也没有加入无实际实现的 `refresh_epochs`。保留同一个 U 的输入/输出投影，不擅自替换为 `U_L K U_R.T`。

## 已实现

- 默认 `dt.basis.mode=joint_gradient, rank=8, compare_dct=true`。只在 fit 轨迹、`t>r` 样本上累计 FP64 的完整 M。做 `M=L diag(s) R.T` 的 SVD，再对左右加权候选拼接 `[L diag(s), R diag(s)]` 做 SVD、压缩到预先指定的秩、QR 和正交验证；没有丢弃反对称部分，也没有仅使用左奇异向量。秩边界间隙过小、有效谱不足或样本不足时拒绝该数据基候选。
- 同时预注册全通道、常数时间 DCT 对照（Kitchen 秩 10）。每个 U 独立重算 R 并拟合 K；tune 数据上的完整梯度风险共同选择 U/K/ridge，每个结构只锁定一个候选，之后才进入独立 accept 验收。8 是一个计算预算选择，不是最优秩结论；没有穷举所有子空间或秩。
- `gradient` 保留旧对称一阶规则作为显式消融，按 sym(M) 的绝对特征值选方向，不再作为默认或唯一方向来源。
- `residual` 提供未中心化的残差二阶矩方向。它不是条件方差 PCA。
- 上述左右联合 SVD 等价于对 `M M.T+M.T M` 取主方向，最大化两侧投影能量，仍不等于优化 `||U.T M U||`，也不含完整回归曲率。默认 family 仍为 diagonal；若需要表达非对称修正，必须显式使用 `dt.family=full`，此时仍与标量、对角和基线比较。
- `dct_all` 对所有动作通道使用相同的时间频率集合；秩必须是动作维数的整数倍，例如 Kitchen 的 10 或 20。拒绝 rank=8 这种截断部分通道的请求。旧 `legacy_first4` 仅作为显式消融保留。
- warmup 的零增益占位 U 使用覆盖全通道的 DCT；数据 U 确定后才计算 `R=H U`、`p=U.T v` 和拟合 K。校准期间不更新模型参数，也不把 U/K 加入优化器。
- U、family、K、global scalar 和归一化坐标以完整 `Correction` 传递。单独更换 U 必须先清零增益；主模型与 EMA 全部通过校验后才安装同一对状态，保存 checkpoint 后继续训练。生命周期诊断传递完整 `(U,K)`，允许不同阶段采用不同秩，并拒绝跨归一化器迁移。
- checkpoint 增加 U/K/family/normalizer 的 SHA-256 完整性校验，支持按保存的 U/K 形状恢复不同秩，不重新抽取方向。旧版缺少指纹的 checkpoint 只接受已知的 `legacy_first4` 基，属于显式兼容路径，不能补做历史完整性证明。
- 独立全空间标量保留 `dt_global*v` 的实现。即使数据 U 退化，预注册 DCT 对照和全空间标量仍可各自拟合、检查和验收；没有有效收益则回退零修正。
- K 保留负增益及完整非对称结构，谱范数上限至多 2；增加 fit 样本上 `||UKU.T v||/(||dw||+1e-8)` 的预注册 0.99 分位数检查，默认上限 2。该数值是可配置的工程保护，不是推导出的安全常数。
- 用 family 元数据进行比较，不再从 `early_gain`、`middle_gain` 等候选名字猜测结构。仍保留独立 tune/accept/report 轨迹，完整原 MP1 损失和配对随机数，记录梯度余弦、范数、预计裁剪及缩放。
- 报告记录选基谱、checkpoint/normalizer 指纹、实际 U、候选 K、动作排列、随机种子、每个基的拟合采样数、固定 Delta 桶计数及桶内/全局矩阵强度、幅度拒绝、选基/拟合/检查耗时和整个校准耗时。计数是随机抽样次数，不是独立轨迹数；没有逐 Delta 桶独立学习 K。

U 和 K 允许在同一 fit 轨迹集上联合拟合；这等价于在拟合集选择更复杂的候选，本身不是数据泄漏。关键是独立 tune/accept/report 轨迹负责选择、验收和最终报告，拟合收益不作为泛化证明。将 U 发现与 K 拟合再拆成两组也合理，但会减少每组可用轨迹，本次未默认添加第五个分区。

## 配置示例

默认运行原来的 DT_MP1 训练命令即可使用数据基。下列覆盖项必须在实验之前确定，避免看 report 结果后再选择并报告同一批结果：

```text
# 全通道常数时间基，Kitchen 单步动作宽度为 10
dt.basis.mode=dct_all dt.basis.rank=10 dt.basis.compare_dct=false

# 完整 K 的联合行列空间消融
dt.family=full dt.basis.mode=joint_gradient dt.basis.rank=8

# 复现旧受限方向基
dt.basis.mode=legacy_first4 dt.basis.rank=8 dt.basis.compare_dct=false

# 同一训练设置的严格零校正对照
dt.family=baseline
```

新实验使用新的运行目录。新配置与旧运行的完整恢复签名不同，不应通过绕过检查将其混接。已有旧模型状态可按上述兼容规则加载；若要从旧 warmup 重新分支，应使用明确的新分支配置及独立输出目录。

## 验证与剩余限制

CPU 回归覆盖 22 项：零 K 的完整梯度与 AdamW 更新、推理不受 DT 影响、非对称 K 的离线/在线一致性、解析 JVP/VJP、有限差分、补充报告的精确反对称反例、曲率改变方向排序、全局 M 为零的反例辨析、joint 候选、全通道 DCT、谱退化、样本不足、不同秩恢复、指纹篡改拒绝、U/K 错配拒绝、归一化器变化拒绝、主模型/EMA 原子安装、整对跨阶段迁移、增益尾部拒绝、全空间标量独立性、双基候选的训练/校准/恢复/生命周期诊断以及已有点云契约。

测试使用 Windows CPU、PyTorch 2.1、小网络和合成数据。没有真实 Kitchen HDF5、服务器定制 robomimic/robosuite 源码和 CUDA rollout，本次不能验证真实点云坐标/裁剪/采样的一致性、GPU 时间收益或成功率。点云的来源记录与训练前 reset/infer/step 检查继续保留，不把 shape 正确说成坐标系已核实。

## 逐文件同步

在本地 PowerShell 执行。目标目录沿用用户已有服务器工程；这里仅列命令，没有代替用户向服务器传输。

```powershell
scp -r "D:\GithubRepo\EfficientFlow\policy\dt_mp1.py" chaoyue@10.181.9.210:/home/chaoyue/MyMP1/EfficientFlow/policy/
scp -r "D:\GithubRepo\EfficientFlow\common\dt_mp1_basis.py" chaoyue@10.181.9.210:/home/chaoyue/MyMP1/EfficientFlow/common/
scp -r "D:\GithubRepo\EfficientFlow\common\dt_mp1_state.py" chaoyue@10.181.9.210:/home/chaoyue/MyMP1/EfficientFlow/common/
scp -r "D:\GithubRepo\EfficientFlow\common\dt_mp1_calibration.py" chaoyue@10.181.9.210:/home/chaoyue/MyMP1/EfficientFlow/common/
scp -r "D:\GithubRepo\EfficientFlow\common\dt_mp1_lifetime.py" chaoyue@10.181.9.210:/home/chaoyue/MyMP1/EfficientFlow/common/
scp -r "D:\GithubRepo\EfficientFlow\workspace\train_dt_mp1_workspace.py" chaoyue@10.181.9.210:/home/chaoyue/MyMP1/EfficientFlow/workspace/
scp -r "D:\GithubRepo\EfficientFlow\config\DT_MP1.yaml" chaoyue@10.181.9.210:/home/chaoyue/MyMP1/EfficientFlow/config/
scp -r "D:\GithubRepo\EfficientFlow\tests\test_dt_mp1.py" chaoyue@10.181.9.210:/home/chaoyue/MyMP1/EfficientFlow/tests/
scp -r "D:\GithubRepo\EfficientFlow\docs\DTMP1_basis_review_20261105_zh.md" chaoyue@10.181.9.210:/home/chaoyue/MyMP1/EfficientFlow/docs/
```
