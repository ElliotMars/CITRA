# 真实数据就绪性与第一组回测结果（2026-10-28）

已完成 DSM/DMX 连续三年真实 AFD/ASOS 的最小预测实验。主实验 39 个独立训练模型，
DLinear 对照 12 个，词袋哈希对照 3 个；全部使用种子 7、17、29。
本轮结果没有显示 Full Text 优于 TS-only，也没有显示当前认证索引带来系统加速。
未开发 Hybrid Index。历史 NumPy 认证、Temporal AVL、ReTaMTSF 和旧实验产物均保留。

## 可用性协议与数据审计

这是 **真实观测/真实文本上的假设回放**，不是获证的历史首次公开时间回测。

- `verified_availability`：要求数值来源已获证，未知文本排除；不接受 simulated 升级。
  本批数据实际拒绝，原因是 `verified_availability requires verified numeric availability`。
- `issuance_asof_assumption`：文本可见阈值为经文件名/WMO/PIL 核对的发行时间加 δ 小时，
  δ∈{0,1,6,24}。仍标记 `simulated`，另外保存显式协议与原始 provenance。
- 数值独立采用观测 valid time +1 小时；小时聚合的可用时刻为小时结束和所有贡献记录假设到达时间的最大值。
  缺失格不补造可用时间。下载时刻、mtime、事件时间不冒充首次公开/历史摄入时间。
- 发行头存在冲突的产品隔离；同一 doc_id 存在无法确定公开顺序的多版本时整组排除。
  原始内容/下载 metadata/来源 URL/检索时刻/SHA256、隔离成员与原因均保留。

来源为 [IEM NWS 产品归档](https://mesonet.agron.iastate.edu/cgi-bin/afos/retrieve.py?help=)
与 [IEM ASOS 观测](https://mesonet3.agron.iastate.edu/cgi-bin/request/asos.py?help=)，逐年请求 UTC 区间。
DSM 对应 DMX 的关系由源站站点元数据核对。每年低于 9999 产品限额，保留产品覆盖所有日历日；
这不证明归档收齐了所有历史发行，更不证明当时可获取归档的当前版本。

|年份|AFD 原始/保留/时间冲突隔离|ASOS 原始行|完整小时网格|温度/风速缺失率|
|---|---:|---:|---:|---:|
|2020|2240 / 2151 / 89|8776|8784|0.0911% / 0.2505%|
|2021|2217 / 2207 / 10|8756|8760|0.0457% / 0.1484%|
|2022|2206 / 2204 / 2|8759|8760|0.0114% / 0.1027%|

共 6663 份原始产品，隔离 101 份，保留 6562 份；精确重复产品、ASOS 重复时间戳组均为 0。
26304×2 数值格中缺失 57 格。数据在 `dataset/citra_real_dsm/`（原文/向量不纳入 Git），
永久审计副本在 [readiness JSON](citra_realdata_readiness.json) 和
[四种延迟审计](../artifacts/citra_real_dsm/availability_delay_audits.json)。
[严格模式拒绝记录](../artifacts/citra_real_dsm/verified_readiness.json) 不能解读为通过。

## 时间划分、文本表示和实验预算

2020 年训练、2021 年验证、2022 年测试；标准化只拟合训练期已可用的数值格。
训练标签必须在训练结束前到达，验证标签必须在验证结束前到达；不使用测试标签选模型或表示。
预测输入 48 小时，目标未来 6 小时的 temperature_F 与 wind_speed_knots。
每 24 小时一个 UTC 预测起点，训练/验证/测试实际窗口数为 **361 / 363 / 363**。
四种延迟、全部种子和方法的预测起点完全相同，汇总脚本逐一比较 cutoff 数组并保存 cohort SHA256。
测试中两天因目标缺失被排除；不会对未知历史值后向填充，只前向填充截止时刻已可用的历史格。
这是日频起点评价，不是全年所有小时起点，也不是多地区结论。

文本表示为冻结 `sentence-transformers/all-MiniLM-L6-v2`，revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`，实际使用 SentenceTransformer 后端，384 维归一化向量。
保留正文和发行头，移除非空白 ASCII 控制字符、合并空白，截断 256 wordpieces。
每条 doc_id/version_id 保存原文/预处理文本哈希、向量行号；保存模型各文件哈希和编码策略。
模型权重 SHA256：`53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db`。
实际模型加载 4.233 秒、预处理 1.223 秒、6562 文档编码 11.003 秒；下载时间独立于编码。
没有 PCA、测试标签拟合或测试集表示筛选；这也不证明编码器预训练语料满足历史部署时点约束。

统一 20 epoch、AdamW lr=.001、batch32、weight_decay=.0001、梯度裁剪1，按最佳验证 MSE 选 checkpoint。
key/impact 维度8、查询隐藏维32、RBF sigma=.7、线性融合范数上限1；配置在读取测试指标前登记。
三种方法及因果控制分别训练，Full/CITRA 推理共享同一份冻结 Full checkpoint。
TS-only 每种种子只训练一次，四种文本延迟复用相同数值结果，训练时间汇总不重复计入。
训练时重新计算可训练 key/impact；只在冻结评估后缓存，不缓存可变训练图。

Shuffled Text Control 使用**从此前合法可见文档有放回采样内容、重新分配到原有时槽**的因果控制，
保留原时槽的可用性和元数据。最初无合法供体时保留自身。不从未来文本池全局置换；
单纯在完整核求和中置换成对 key/impact 不会改变结果，因此不将其伪装为有效控制。
这是一种可复现的历史前缀控制，不等于无条件随机文本或完全去除季节相关性。

## 实际预测结果

以下为训练期标准差标准化空间的联合 MSE/MAE，三种种子均值 ± 样本标准差（ddof=1）。
种子标准差只反映训练随机性，不是天气事件、年份或地区不确定性的置信区间。

|方法，线性骨干，δ=0|标准化 MSE|标准化 MAE|
|---|---:|---:|
|TS-only|0.385913 ± 0.006282|0.408828 ± 0.006642|
|Semantic Top-K (K=8)|0.403327 ± 0.003134|0.419637 ± 0.002563|
|Full As-of Text|0.701655 ± 0.080980|0.601491 ± 0.040110|
|Shuffled Text Control|0.619095 ± 0.036130|0.564782 ± 0.021861|

|文本延迟|TS-only MSE|Semantic Top-K MSE|Full Text MSE|Shuffled MSE|
|---|---:|---:|---:|---:|
|0h|0.385913 ± 0.006282|0.403327 ± 0.003134|0.701655 ± 0.080980|0.619095 ± 0.036130|
|1h|同一数值结果|0.407714 ± 0.006161|0.702174 ± 0.081312|0.619504 ± 0.036425|
|6h|同一数值结果|0.403577 ± 0.004600|0.702732 ± 0.081366|0.619453 ± 0.036472|
|24h|同一数值结果|0.400918 ± 0.008465|0.705732 ± 0.083067|0.621398 ± 0.037887|

延迟变化对这次 Full 的影响较小，没有扭转负结果。各延迟的 MAE、每一种子数值、最佳 epoch、
训练成本、每目标物理单位指标均见 [linear_summary.json](../artifacts/citra_real_dsm/linear_summary.json)
与 [CSV](../artifacts/citra_real_dsm/linear_scores.csv)。原始物理联合 MSE 混合 °F 与 knot，不能当作有单一单位的误差。
零延迟 TS-only 的温度 MSE/MAE 为 32.5299 °F² / 4.2701 °F，风速为 16.1017 knot² / 2.9513 knot；
Full 对应为 85.6832 / 7.4887、27.8916 / 4.0666（均为种子均值）。

主实验出现负结果后，固定零延迟与相同预算，加入 25 点历史窗口平滑的 DLinear 分解线性候选。
这是本项目实现的数值对照，不声称复刻官方模型全部参数，也不预设一定更好。

|方法，DLinear，δ=0|标准化 MSE|标准化 MAE|
|---|---:|---:|
|TS-only|0.394271 ± 0.006653|0.392789 ± 0.003863|
|Semantic Top-K|0.401070 ± 0.013413|0.399727 ± 0.006182|
|Full As-of Text|0.585591 ± 0.127370|0.516518 ± 0.069309|
|Shuffled Text Control|0.507023 ± 0.067621|0.477758 ± 0.037496|

DLinear 改善了 TS-only MAE，但未改善联合 MSE；不能宣称它在本实验全面更强。
Full 在第二种骨干上仍不优于 TS-only，控制也没有出现“打乱后失去原有收益”的现象。
保留并实际运行相同384维词袋哈希 Full 消融，δ=0，MSE **0.668771±0.098475**、MAE **0.588501±0.051493**，
也未优于 TS-only。DLinear 不再扩展四种延迟，哈希只比较 Full，不将这两个额外对照说成完整矩阵。
哈希读取整篇原文，MiniLM截断256 wordpieces，不能把两者差异单独归因为语义编码质量；长AFD的截断影响尚未做受控实验。
详见 [DLinear](../artifacts/citra_real_dsm/dlinear_summary.json)、[hash](../artifacts/citra_real_hash/linear_summary.json)。

## 计时口径和真实查询诊断

最终计时入口为 `profile_all`：单进程顺序评估 TS-only、Top-K、Full/CITRA，没有训练混入。
预先固定第一种种子7、文本延迟0，均匀抽取12个测试起点，每点重复3次，保留冷调用，不挑最快一次。
准确率另以全部363窗口批量计算，不能把批量准确率耗时当成单次延迟。
CPU/GPU 分开保存；这是小样本诊断而非生产吞吐量测试，GPU 的短算子/同步延迟波动显著。

TS-only 不加载文档特征、不构建文本索引、不执行 Query Encoder；相关在线/离线字段为0。
Top-K 与 Full 使用向量化合法 mask 和核聚合，不要求 AVL。只有认证分支显式离线建索引。
逐查询分别记录 Query Encoding、Exact/Approximate Aggregation、输入/查询/聚合/输出传输、
Forecast Inference、输入验证、模型字节版本校验和 End-to-End。GPU 各阶段同步；版本校验含参数拷贝/哈希，
计入总延迟但不冒充纯模型计算。`index_query_seconds` 是聚合别名，`data_transfer_seconds` 是传输分项之和，不能重复相加。
基础 MiniLM 编码从数据 manifest 报告，不把离线值塞入在线查询；未测量字段保持 null，不伪装成0。

Index Construction 单列；Index Maintenance 用独立存储重放最后100条真实文档到达，冻结 key/impact，
单列节点创建、文档读取、纯索引时间、维护总时间和分位数。这批数据没有可验证的修订维护案例，
不声称已覆盖真实 correction 成本。Full/CITRA 共用冻结参数及版本摘要，不分别训练近似模型。

|方法|CPU P50 / P95 (ms)|A100 P50 / P95 (ms)|
|---|---:|---:|
|TS-only|0.289 / 0.335|8.271 / 11.231|
|Semantic Top-K|1.121 / 1.194|1.965 / 3.303|
|Full As-of|1.223 / 1.873|13.846 / 16.715|
|CITRA ε=.01|121.358 / 246.509|133.826 / 261.123|
|CITRA ε=.1|100.509 / 136.707|113.577 / 156.539|
|CITRA ε=1|100.406 / 114.441|108.007 / 115.897|

GPU TS-only 反而慢于 Top-K，表明本机短调用同步/调度状态影响明显，不能用该表给不同预测模型排速度名次。
CPU 对照与 GPU 均显示当前认证树慢于同参数向量化 Full；没有选择最有利的设备/测次宣称优势。

下面为 **CPU 各阶段均值**（ms），避免把不同阶段的 P50 相加；同设备同一线程配置。
CPU 的“数据传输”字段主要是张量/NumPy 转换，不表示发生了 GPU 传输。

|方法|Query Encoding|Exact / Approx Aggregation|数据转换|Forecast|版本校验/输入验证/其他|End-to-End|
|---|---:|---:|---:|---:|---:|---:|
|TS-only|0|0 / 0|0.051|0.046|0.202|0.300|
|Top-K|0.180|0.372 / 0|0.089|0.063|0.440|1.144|
|Full|0.203|0.548 / 0|0.099|0.070|0.491|1.411|
|CITRA .01|0.207|0 / 135.177|0.113|0.081|0.494|136.072|
|CITRA .1|0.198|0 / 104.012|0.107|0.077|0.475|104.869|
|CITRA 1|0.199|0 / 88.853|0.108|0.077|0.468|89.705|

本次 CPU/GPU Index Construction 为4.597/4.737秒，100次维护总耗时0.0789/0.0822秒；
CPU 单次维护 P50 0.768ms。建树/维护都发生在 CPU，设备标签只表示上游预测模型所在设备。
完整未取整分项、逐查询三次计时、节点/维护记录及计时代码哈希分别见
[CPU](../artifacts/citra_real_dsm/profiles_cpu/) 与 [GPU](../artifacts/citra_real_dsm/profiles_final/)。
此前 `linear_profile*`/`linear_*_profile_isolated` 为保留的诊断尝试，不纳入上述最终表：
早期存在 GPU 测量重叠，随后将输入验证从纯输入传输计时中分离，最终以新口径顺序重测；没有删除旧产物。

真实学习到的索引共有2287个节点、6562个版本区间，半径 P50/P95/max 为0.097777/0.173551/0.277508，
Python/NumPy resident estimate约14.55 MB（非进程峰值内存）。12个实际查询的扫描占合法可见文档比例：

|输出 L2 预算 ε|平均扫描比例|最大实际输出偏差|完全精确退化次数（12次查询）|界细化总次数|
|---|---:|---:|---:|---:|
|.01|94.3764%|0.000464|0|2245|
|.1|76.9846%|0.003821|0|931|
|1|73.3540%|0.006110|0|0|

预算单位为单样本 `[6,2]` 标准化预测的 L2，不是 MSE。每预算12组与精确 Full 的输出误差/证书检查通过，
共108次计时查询；非形式化 IEEE 保证，float64 对照 atol/rtol=1e-10。
平均扫描少于100%不等于每次都大幅省读：ε=1时扫描范围为1.28%–97.52%。
离线另抽192个完全可见节点，保存局部 Hessian 上界、全局界、实际补偿求和偏差、半径/距离；
本次局部聚合界/实际误差中位比约9.44，局部界/全局界中位比约0.270。
这些 oracle 诊断不参与停止条件，也不计入在线延迟。

## 研究判断与失败项

1. **Full Text 不优于 TS-only**：本地区、本预算、两种骨干均为负结果；不能据此证明所有文本模型无用。
   不截断完整合法求和来制造“Full”收益。
2. **未获得文本语义收益证据**：打乱控制反而更好但仍差于 TS-only；MiniLM 不优于哈希的这次 Full 消融。
   可能存在优化困难、无限累积历史文本与逐年文档数量增长等问题，尚未做因果归因或调参验证。
3. **可避免部分全扫描，但没有速度收益**：宽预算下完全没有因界超标触发细化，仍平均扫描73.35%。
   ε=1时访问节点中约80.68%为时间混合节点；混合叶必须逐条检查，形成主要的扫描下限。
4. **当前耗时主要落在 Python 树遍历/小块聚合**，精确对照已向量化；紧预算额外增加界细化。
   局部界仍偏松，但宽预算下不细化也很慢，不能仅把性能问题归因于误差界。
   只有一种词典序 AVL 空间划分，缺少受控空间划分消融，因此不能排他证明空间划分的贡献。
5. **目前不足以支持复杂 Hybrid 开发**：时间混合给出结构优化动机，但预测目标尚无文本收益，
   6562份文档也不足以代表大规模系统；优先验证训练稳定性、历史累积效应、更多年份/地区和可用性证据。

尚未通过：历史首次公开时间认证、全小时起点/多地区推广、模型族充分调参、强骨干全面优越性、
大规模速度优势、正式机器数证书、Hybrid/ICDE 创新性证明。源站下载曾遇到断连/截断，有限重试和官方镜像后完成；
101条头信息冲突未修补为“正确”发行时间。哈希训练后的汇总曾因字符串 Path 类型报错，修复后重新汇总，未重新选择模型。

## 复现命令与产物

本机 Python `/root/miniconda3/envs/text4ts/bin/python`，NumPy2.2.5、PyTorch2.4.1/CUDA12.1、A100 80GB。
正常 PyTorch 环境直接执行以下命令；本机用过程级 `LD_PRELOAD=/tmp/citra-ittapi/libjitprofiling.so`
修复 iJIT 导入问题，官方 ITT 的固定提交/构建命令保留在 [历史阶段记录](citra_progress.md)。
编码额外依赖 `requirements-citra-text.txt`；本机隔离安装到 `/tmp/citra-textdeps` 与 `/tmp/citra-encoder-minimal`，
编码时使用对应 PYTHONPATH，不修改全局依赖。

以下准备命令针对不存在的输出目录；已有目录会拒绝覆盖。新机器需先下载，若来源档案变化，应核对保存的哈希，不能假定再次下载逐字节相同。

```bash
python -m pip install -r requirements-citra.txt -r requirements-citra-text.txt
for citra_year in 2020 2021 2022; do
  citra_next=$((citra_year + 1))
  python scripts/download_nws_afd.py --office DMX --server mesonet3.agron.iastate.edu \
    --start "${citra_year}-01-01T00:00Z" --end "${citra_next}-01-01T00:00Z" \
    --output "dataset/citra_real_dsm/raw/AFDDMX_${citra_year}.zip"
  python scripts/download_asos.py --station DSM --server mesonet3.agron.iastate.edu \
    --start "${citra_year}-01-01T00:00Z" --end "${citra_next}-01-01T00:00Z" \
    --output "dataset/citra_real_dsm/raw/DSM_${citra_year}.csv"
done
python scripts/build_citra_multiyear.py collect --raw dataset/citra_real_dsm/raw \
  --years 2020 2021 2022 --protocol issuance_asof_assumption --numeric-delay-hours 1 \
  --output dataset/citra_real_dsm/collection
python scripts/build_citra_multiyear.py numeric --collection dataset/citra_real_dsm/collection \
  --output dataset/citra_real_dsm/numeric_only
python scripts/encode_citra_text.py --documents dataset/citra_real_dsm/collection/documents.jsonl \
  --model sentence-transformers/all-MiniLM-L6-v2 --revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 \
  --device cuda:0 --batch-size 64 --max-length 256 --output dataset/citra_real_dsm/minilm
python scripts/build_citra_multiyear.py prepare --collection dataset/citra_real_dsm/collection \
  --features dataset/citra_real_dsm/minilm --output dataset/citra_real_dsm/prepared
python scripts/build_citra_multiyear.py hash --collection dataset/citra_real_dsm/collection \
  --hash-dimensions 384 --output dataset/citra_real_dsm/hash384
```

本机编码使用相同 revision 的 Hugging Face 缓存 `--local-model-path`，权重和全部文本向量哈希均在 manifest 中。
实际训练/评价入口如下，目录改为新的输出路径可重跑；注册文件与完成结果允许同配置断点跳过，不强制覆盖未完成目录。

```bash
python scripts/run_citra_real_experiments.py train --config configs/citra_real.yaml \
  --output artifacts/citra_real_dsm --device cuda:0
python scripts/run_citra_real_experiments.py train --config configs/citra_real_dlinear.yaml \
  --backbone dlinear --output artifacts/citra_real_dsm --device cuda:0
python scripts/run_citra_real_experiments.py train --config configs/citra_real_hash.yaml \
  --modes full_asof_text --output artifacts/citra_real_hash --device cuda:0
python scripts/run_citra_real_experiments.py profile_all --config configs/citra_real.yaml \
  --output artifacts/citra_real_dsm --profile-output artifacts/citra_real_dsm/profiles_final --device cuda:0
python scripts/run_citra_real_experiments.py profile_all --config configs/citra_real.yaml \
  --output artifacts/citra_real_dsm --profile-output artifacts/citra_real_dsm/profiles_cpu --device cpu
python scripts/validate_citra_timestamps.py --data dataset/citra_real_dsm/prepared \
  --availability-protocol issuance_asof_assumption --text-delay-hours 0 --seq-len 48 --horizon 6 --stride 24
python scripts/validate_citra_timestamps.py --data dataset/citra_real_dsm/prepared \
  --availability-protocol verified_availability --stride 24
```

最后一条实际预期退出2，严格模式失败是已记录限制，不把它当作假设回测通过。
四种延迟审计在相同 `audit(..., stride=24, text_delay_hours=δ)` 入口执行。
原始下载/编码/数值文件留在本机数据目录；每次训练保存 resolved config、Scaler、全日志、最佳 checkpoint、
363份预测/真实目标与 cutoffs；所有种子明细和汇总在两个 artifacts 目录，不仅保存最优种子。

实际回归命令：

```bash
PYTHONDONTWRITEBYTECODE=1 LD_PRELOAD=/tmp/citra-ittapi/libjitprofiling.so \
  /root/miniconda3/envs/text4ts/bin/python -m unittest discover -s tests -v
```

67项测试通过；覆盖原认证/版本/AVL 测试、显式假设不升格、因果供体、延迟不改变数值 cohort/Scaler、
特征内容绑定、TS-only 无 Query/Index 工作、延迟构建索引与 DLinear 可训练性。实际真实数据诊断另外验证共享参数证书。
测试日志、来源/结果完整性与环境摘要见 artifacts 中的本轮记录。
测试日志：[67项回归](../artifacts/citra_real_dsm/tests.txt)。另执行
`python scripts/verify_citra_real_results.py --output artifacts/citra_real_dsm/result_verification.json`，
独立重算63行汇总所对应的54组预测指标，核对12个原始/metadata文件哈希、6562份特征、363个测试起点/目标、
CPU/GPU共享模型摘要与72组输出证书；
[结果一致性验证](../artifacts/citra_real_dsm/result_verification.json) 通过。
这里的 `verified=true` 仅表示产物一致性，不表示历史可用性获证。
另外以相同配置重新训练seed7的零延迟Full模型20个epoch，363份预测与原运行逐元素一致，最大差为0；
MSE均为0.6833669126048277，见 [复现核对](../artifacts/citra_real_dsm/reproducibility.json)。
这次重复不计为第四种种子；数值专用bundle也已由新增准备入口重建，fingerprint与Scaler一致。

## 修改文件索引

|范围|文件|
|---|---|
|协议/天气接口|`citra/availability.py`、`data_provider/citra_weather.py`、`scripts/prepare_citra_weather.py`、`scripts/validate_citra_timestamps.py`、`scripts/download_nws_afd.py`|
|多年份与特征|`scripts/build_citra_multiyear.py`、`scripts/encode_citra_text.py`、`requirements-citra-text.txt`、`dataset/citra_real_dsm/.gitignore`|
|模型/训练/准确率|`models/CITRA.py`、`citra/learning/numeric_backbones.py`、`citra/learning/training.py`、`exp/exp_CITRA.py`、`run_CITRA.py`|
|冻结推理/索引诊断|`citra/learning/inference.py`、`citra/learning/diagnostics.py`、`citra/temporal.py`、`citra/index.py`|
|实验/复核|`scripts/run_citra_real_experiments.py`、`scripts/verify_citra_real_results.py`、`configs/citra_real{,_dlinear,_hash}.yaml`、`tests/test_citra_real_protocol.py`|
|文档/产物|`README.md`、`docs/citra_weather.md`、`docs/citra_progress.md`、本报告和readiness JSON、两个新增artifacts目录|

用户当前的`prompt.md`修改保持原样。未提交或推送，不覆盖旧权重、数据或历史实验。
