# 最终可行性诊断（2026-10-29，当前阶段）

基于`78af29e`、`feature/citra-prototype`，先审计并登记最终协议，再执行系统与内容价值两条独立诊断。
核心模型、聚合Oracle、冻结推理、Temporal AVL与ReTaMTSF代码不变，不开发Hybrid，也不推送。

真实2020–2023全部1451个预测起点：七天合法文档数均值41.941、P50/P95/P99/Max=42/44/46/47，空窗口为0。
同checkpoint、query/Key/Impact、归一分母与缓存，24个固定真实query×10次热计时，另保留冷调用。
主结果只装载cutoff当时已公开的前缀库：NumPy Exact端到端P50/P95/P99/Max为
0.864/0.975/1.114/1.588ms，AVL(ε=.1)为2.707/2.965/3.854/11.245ms；
纯向量化聚合P50约0.035ms。原始全归档回放也保留，避免以未来记录树结构夸大缺点。
ε=.01/.1/1均无误差细化，平均仍扫描95.8%（按文档加权更高），主要是时间混合造成的强制下钻。
System No-Go，不把树遍历/小块fsum总成本排他归因为某个单一因素。

Metadata-only、Content-only、Constant Text、Null/no-metadata各3种种子，共12个新模型，固定原TS骨干/零头/预算。
真实文本、重训因果打乱和TS复用原权重；同权重打乱另评价。非内容输入经过白名单和禁止读取embedding/text的测试。
2022/2023均为已见解释性年份；方案登记后取得新2024，全部权重锁定后才计算其366个共同日期预测。
2024三种子标准化MSE：TS0.309695、真实0.308067、元数据0.307816、Content-only0.308539、
Constant0.308377、Null0.308837、同权重打乱0.308240、重训打乱0.308369。
主指标相对所有对照的7/14日块Bootstrap区间均跨0；部分气温指标真实文本反而更差。
统计先按日期对训练种子平均，不把同一天气目标视为三倍独立观测。Prediction Evidence Insufficient。

合成N=50/500/5000/50000/100000、同质/混合两场景全部10组执行。
同质场景在已测N=5000起有ε=.1交叉，N=100000时Exact/AVL约8.394/0.242ms；
但混合归档未有优势，且真实任务只有42篇。高斯向量、随机时间/5%修订明确标为synthetic，不作为可信真实应用。
83项回归（新增8项）通过；独立核对21份权重、20个来源文件、72组预测、42个主区间、3374次真实性能调用及500次合成查询通过。

最终建议 **No-Go for Current CITRA**，停止当前论文主线，保存时态语义、Oracle、认证/冻结绑定、数据与复现成果。
三份最终报告：
[系统](citra_final_system_diagnosis.md)、[语义](citra_final_semantic_diagnosis.md)、[决策](citra_final_go_nogo.md)。
产物`artifacts/citra_final_diagnosis/`包含注册/环境/哈希、全记录、控制模型、配对日期损失、CI、曲线、命令、异常和测试。
用户prompt与全部旧模型/数据/报告保留；以下为历史阶段记录。

---

# 预测退化诊断与最小残差修正（2026-10-28，历史阶段）

按当前prompt先完成未修改模型的冻结审计与分布图，再实现受控消融。
旧Linear/DLinear各3种种子、每模型1087个训练/验证/诊断窗口：N从约1073增至5457，
Linear文本修正/数值预测范数比从约0.25–0.41增至1.03–1.30；
原Full关闭文本仍弱于独立TS，表明聚合尺度和数值分支退化都需处理。

增加合法可见计数归一、公开时间7/30/90天窗口、固定TS骨干零头残差、历史条件有界门控；
保留原Sum，所有新算子均有独立snapshot补偿求和Oracle和相同分母/窗口的认证语义。
11个配置×3种种子，3个原Full复用、30个新训练；全部训练/验证曲线、逐目标MSE/MAE和所有种子均保存。
加入尾部256-token表示、因果打乱及text-off。仅按2021三种子平均验证MSE锁定
Window7 Count Mean Residual，2022明确标记为已见诊断，不再用它替换选择。

追加2023真实归档、隔离1份发行冲突/2份不明修订及1组数值冲突，原始文件不修改。
旧年度数值/可用性/文本向量前缀和Scaler严格保持，最终364个未参与选择的日频预测起点。
2023标准化MSE：TS **0.301148±0.003939**，所选残差 **0.298032±0.001625**；
但同权重推理打乱仍为 **0.298193±0.001646**，保留约95%的改善，重训打乱也改善TS。
2022所选残差仍略差于TS。可确认退化缓解，尚不能确认可靠文本内容/时间对齐收益。
候选算子为公开时间窗口+合法计数归一核和，对复杂索引/Hybrid继续给出 **No-Go**。

75项回归测试通过；7种算子/门控组合的63组真实Exact/Approx/独立Oracle证书回归通过，
门控有效L由实际g(x)、W范数和相同分母计算，参数变化强制新会话重建。
原NumPy、AVL、ReTaMTSF、性能记录和历史产物保留，未开发新索引。
详细证据、限制、运行成本、失败记录与命令见 [退化诊断报告](citra_failure_diagnosis.md)。
产物为 `artifacts/citra_failure_diagnosis/`，未提交/推送；以下保留之前阶段的历史记录。

---

# 真实三年回测与计时修正（2026-10-28，历史阶段）

已按当前 prompt.md 完成任务1–3的最小真实实验，再完成共享冻结参数的性能诊断。
DSM/DMX 2020–2022年逐年下载，6663份AFD中隔离101份发行头冲突，保留6562份；
26304小时×2列中缺失57格，原始来源、SHA256、重复/缺失/逐日覆盖审计均已保存。
采用明确的 `issuance_asof_assumption`，文本延迟0/1/6/24小时，数值另设观测+1小时；
`verified_availability` 实际拒绝，首次历史公开时间仍未获证，没有把假设回放升级为严格认证。

冻结 MiniLM revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`，
离线逐 doc_id/version_id 保存384维向量、原文/预处理哈希与模型文件哈希。
2020训练/2021验证/2022测试，48→6小时、每日UTC起点，实际361/363/363窗口，训练期单独拟合Scaler。
三种种子7/17/29全部报告；主实验39模型、DLinear12模型、哈希Full消融3模型。
预测起点/目标相同；TS-only跨文本延迟复用不重复计费，文本打乱使用无未来供体的前缀采样控制。

零延迟线性骨干的标准化MSE：TS-only **0.385913±0.006282**、Top-K **0.403327±0.003134**、
Full **0.701655±0.080980**、Shuffled **0.619095±0.036130**。
Full延迟24h为0.705732±0.083067；DLinear仍无文本收益，TS-only的MAE改善但MSE未改善。
这些均为真实观测/文本的显式假设回测，不是fixture，也不是已验证历史首次可用时间的结果。

TS-only不再加载文本特征、编码查询或建文本索引。Full/Top-K向量化精确聚合，
CITRA独立计建树/维护，逐阶段记录编码、传输、聚合、预测、校验、总耗时。
12个固定真实查询×3次计时：ε=.01/.1/1平均扫描94.38%/76.98%/73.35%，未完全退化为精确；
但CPU Full/CITRA(.1) P50 **1.223/100.509ms**，GPU **13.846/113.577ms**。
GPU短调用延迟波动明显，两设备均没有认证树加速证据。实际192节点局部界与36组输出证书对照通过。
宽预算不细化仍大量扫描，说明时间混合造成扫描下限；Python树聚合耗时远大于向量化精确求和。
文本收益与大规模证据不足，本轮不开发Hybrid，不声明ICDE创新或系统加速。

代码涉及 availability、weather adapter、冻结文本编码/多年份准备、训练/推理、DLinear、
真实矩阵/汇总与顺序profiling入口，详情与实际命令/失败项见
[就绪性报告](citra_realdata_readiness.md)。原NumPy数学认证、Temporal AVL、ReTaMTSF、全部历史产物和用户prompt修改保留；未提交/推送。
67项回归测试通过，新增10项覆盖协议、因果控制、特征绑定、数值路径与分解骨干。
最终独立复核63行汇总/54组预测、6562份特征、12个来源文件与72组CPU/GPU证书通过；
seed7 Full另行重复20epoch，全部363份预测与原运行逐元素一致（最大差0）。
真实结果、配置、训练日志、checkpoint、预测数组在 `artifacts/citra_real_dsm/` 和 `artifacts/citra_real_hash/`。

以下保留阶段A历史记录，其中“未运行真实预测”“未评测MiniLM”等为当时状态。

---

# 阶段 A：可训练预测闭环与最小真实适配（2026-10-28）

本轮从 prompt.md 452 行开始，按最后要求先落实第一阶段。新增真实可训练的 PyTorch 模型和
Point-in-Time 天气接口；CPU/A100 的集成训练已实际执行。真实天气预测实验因历史可用性缺证而未运行。
Hybrid、100K 系统实验和真实 Accuracy–Latency 曲线尚未完成，不把第一阶段结果当作这些里程碑通过。

## 新增/修改文件与实际功能

- models/CITRA.py、citra/learning/{temporal_encoder,text_encoder,influence_encoder,fusion,training,inference}.py：
  历史/目标/horizon 条件查询，冻结基础特征上的可训练 key 与带符号影响，Torch 完整合法求和，线性认证头，
  版本化冻结缓存与显式失效检测。
- data_provider/citra_weather.py：UTC 数值可用性、仅训练拟合 Scaler、按时间窗口/模型拟合边界划分、
  文档版本 mask、可复现 fixture；MOS 发行时间/有效时间 join 扩展函数。
- scripts/download_nws_afd.py、download_asos.py、citra_download.py、prepare_citra_weather.py、
  validate_citra_timestamps.py：实际下载、原始 hash/来源保留、WMO/文件名交叉核对、按小时聚合与严格审计。
- exp/exp_CITRA.py、run_CITRA.py、configs/citra.yaml、requirements-citra.txt：统一模式、checkpoint、
  配置/训练日志、MSE/MAE 的真实数据分支、在线/离线成本。合成 fixture 不输出真实 MSE/MAE。
- tests/test_citra_learning.py、test_citra_weather.py：可训练梯度、冻结一致性、错误非线性头、未来信息隔离、
  数据校验与模型拟合边界；保留原 37 项测试。
- README、研究文档、docs/citra_weather.md；artifacts/citra_stage_a 保存模型、配置、fixture、
  全部训练/推理记录和三个地区的真实下载样本，不覆盖任何旧数据/权重。

## 实际测试与训练

`PYTHONPATH=/tmp/citra-torch-cpu /root/miniconda3/envs/text4ts/bin/python -m unittest discover -s tests -v`：
56 项通过，13.315 秒；原 test_certified_bound 的 1296 组参数仍保留。日志 citra_stage_a_tests.txt。
CPU 和 A100 分别完成 ts_only/semantic_topk/full_asof_text，各 3 epoch，保存最佳验证 checkpoint。
每模型评估 18 个 test 窗口；Full/CITRA 每设备做 54 次共享参数证书对照，预算检查全部通过。
在文件路径重定位后也实际验证 checkpoint 可重新加载。数值范围为明示 float64 容差，非形式化机器数保证。

结果：CPU Full/CITRA(epsilon=.1) P50=1.558/1.829 ms；GPU=3.075/3.157 ms。
所有预算累计扫描 819 条合法文档，返回零截断界，CPU/GPU 最大舍入偏差约 1.12e-16/2.80e-17。
这证明了完整预测执行链可运行，没有证明近似节省、文本预测增益或总体系统优势。

## 环境和复现

原 text4ts PyTorch 2.4.1 因 iJIT_NotifyEvent 导入失败。CPU 使用 /tmp/citra-torch-cpu 的官方 2.5.1+cpu
隔离安装，未改全局环境。GPU 使用现有 2.4.1/CUDA 12.1，过程级加载 Intel 官方 ITT 实现后运行 A100。
来源：[PyTorch 问题记录](https://github.com/pytorch/pytorch/issues/123097)、[Intel ITT API](https://github.com/intel/ittapi)。
没有编造缺失符号、替换数学运算或伪装 GPU 执行。已验证的兼容构建命令如下：

```bash
git clone https://github.com/intel/ittapi.git /tmp/citra-ittapi
git -C /tmp/citra-ittapi checkout 54835bb96f1b85f8f21841ce821487f69d95ecce
gcc -shared -fPIC -I/tmp/citra-ittapi/include /tmp/citra-ittapi/src/ittnotify/jitprofiling.c \
  -ldl -o /tmp/citra-ittapi/libjitprofiling.so
LD_PRELOAD=/tmp/citra-ittapi/libjitprofiling.so /root/miniconda3/envs/text4ts/bin/python run_CITRA.py \
  --fixture --mode all --epochs 3 --device cuda:0 --output /tmp/new-citra-gpu-run
```

正常可导入的 PyTorch 环境不需要此本机兼容步骤。CPU 命令见 citra_experiments.md。
下载 MKL 的另一路修复尝试较慢，已停止本轮自己的后台下载，未覆盖环境或其他进程。

## 真实数据、未通过的校验与研究边界

已实际获取 DSM/DMX、ORD/LOT、IAH/HGX 2020-01-01 至 01-03 的 12/16/12 份 AFD、各 48 小时 ASOS；
源站元数据确认办公室对应。网络曾重置/截断，已添加有限重试，ORD 经官方镜像下载。
实际 ZIP 格式也已核实并交叉检查 WMO/PIL；没有仅凭文件名猜测历史可用性。

归档只提供发行/观测时间，逐条首次可用/历史摄入证据仍未知。严格校验真实样本实际退出 2，
strict_backtest_ready=false，这是预期的失败关闭；不能报告真实 MSE/MAE 或年份训练已经完成。
原始数据、来源、SHA 和拒绝原因见 artifacts/citra_stage_a/weather_probe。
可用性 ledger 导入和显式模拟延迟路径已实现，但没有伪造 ledger 以让真实实验通过。

认证前提仍是合法完整求和、固定且一致的 z/v/模型、内置线性头与真实时间来源。
不保证真实未来 MSE；没有 IEEE 总舍入证明。对不受支持的头/过期缓存拒绝认证。
目前只有闭环工程增量，尚不能声称 ICDE 研究贡献成立。

下一步优先补齐历史可用性证明（或前瞻采集真实接收时间）并开展按年份划分的真实预测；
随后按任务第二阶段保留 AVL、实现独立 Hybrid/成本感知查询，以实际 encoder 的近距查询测量混合节点瓶颈。
本轮没有用远距离核趋零、未测的大规模结果或真实预测收益来替代这些待完成工作。

## 上阶段审查修复交付记录（保留）

本轮按 prompt.md 第 357 行后的代码审查，落实四个 P0 机制修复。用户编辑的 prompt.md 保留；
仍在 feature/citra-prototype，未提交或推送。本轮没有改写原 ReTaMTSF 模型/加载器/训练入口。

## 修改与实现

- citra/index.py：叶级文档引用，内部合并 V/M/S、包围半径和时间极值；可局部更新的 immutable AVL；
  aggregation-only API；局部/全局界及 coarse ablation；完整驻留内存估算。
- citra/temporal.py：按 activation/版本优先级推导互斥可见区间，写入/修订差分维护 AVL；
  新 cutoff 不扫描全集建树；独立 snapshot/oracle 保留，indexed=False 用于全扫描基线。
- citra/kernel.py：球距离区间的严格解析 Hessian 上界、径向极值、对数空间及下溢保护。
- citra/prediction.py：从实际冻结权重自动导出 L；不接受外部 Lipschitz 数值；
  参数/文档版本绑定和过期证书检测。
- citra/synthetic.py、__init__.py、run_citra_synthetic.py：适配新 API 和无索引基线。
- 新增 run_citra_benchmark.py、tests/test_citra_review.py；更新原测试、README 和研究文档。
- 新结果 citra_review_smoke_results.json、citra_review_benchmark.json；
  第一版 JSON 和数据审计结果均保留。

P0-1 已有真正跨 cutoff 的时间可见子树与局部矩维护，非快照失效重建。
P0-2 已移除任意调用端 L；P0-3 已使用 q-dependent 有效局部界；
P0-4 已移除内部节点重复引用/版本列表，并报告完整内存诊断。
这些是机制正确性修复，不等于实际系统贡献已经成立。

## 实际运行结果

CPU 回归：37 项通过，保留 1296 组主证书参数矩阵，并加入随机乱序版本/过期/迟到对照。
最终运行 12.053 秒，日志保存于 citra_review_tests.txt；另验证时态索引 oracle 必须指定 cutoff，
以及 kernel/可见性策略不能在已维护索引上悄悄更换。
预测 smoke：12 组，预算和输出界检查通过；固定未训练头，不报告真实预测准确率。
滚动系统实验：256/4096 文档，3 cutoff × 3 epsilon × 2 查询距离，共 36 组；
每种方法重复 3 次，每规模 8 次修订；聚合界检查和更新后 exact 对照通过。

以下为已保存 benchmark 的实际诊断数据，单位秒/字节：

|文档数|叶引用数|内部引用数|索引含文档驻留估算|tracemalloc 构建峰值|无追踪建库重放|CITRA 建库+查询+更新|Full Scan 建库+查询+更新|
|---|---|---|---|---|---|---|---|
|256|276|0|472222|1415646|0.1960|0.3803|0.1749|
|4096|4412|0|7704973|8473999|3.7330|6.7130|2.7715|

多出的叶引用是历史版本区间，每条区间仅存一次；不是同一记录在多层节点重复。
每规模 8 次修订最多创建 24/34 个节点，最多读取 24/22 个叶桶文档；
平均单次更新 1.555/2.316 ms。更新包含该文档元数据重推导和局部索引维护。
基准中明示追踪器测内存与无追踪计时的区别；RSS、P95 和所有样本见 JSON。

4096 文档、cutoff=30、epsilon_aggregate=0.1：
q=0 时 global/local 都精确扫描 4060 条，local P50=118.070 ms，Full Scan P50=73.928 ms。
q=50 时 global 扫描 4060、访问 1547 节点；local 扫描 2195、访问 1135 节点，
local P50=59.322 ms，Full Scan P50=91.035 ms。
局部界减少远查询细化，但时间版本混合叶仍需处理；整体成本 6.7130 秒高于全扫描 2.7715 秒。
本结果不支持总体加速主张，也不能从减少内部引用推出总内存更低；新增摘要字段有实际存储成本。

## 环境、数学假设与未完成项

Python 3.10.0 / NumPy 2.2.5 的 CPU 原型可运行，未安装新依赖。
PyTorch 复核仍失败：libtorch_cpu.so undefined symbol: iJIT_NotifyEvent；未运行 PyTorch/GPU 集成。
本地真实 Time-MMD 数据为空，未下载/导入 NWS AFD、ASOS、MOS，也没有真实 MSE/MAE。
不存在未通过的 CPU 单元测试；环境导入失败不是通过的实验。

证书只在完整合法求和、固定参数/影响向量、有效时间来源和解析余项范围下成立。
float64 采用声明容差，尚无完整 IEEE 舍入证明，certificate_scope 已明确限定。
局部界过松、时间碎片、保守半径、异质影响和 Python 对象成本仍可导致退化全扫描。
历史回填改变当前系统对过去的认识；严格复现仍需固定 manifest 和实际/未知摄入时间。

本轮已有可验证的时间索引和查询相关细化，但总体性能/真实预测增量仍未建立。
AVL、泰勒和矩合并属于已有构件，不能据此宣称 ICDE 创新性。
Milestone 2 的核心查询/更新和原型系统对照已推进，距离强数据库基线/大规模评估仍有差距。
Milestone 3/4 仍未完成：没有可训练多模态模型、真实适配器或真实预测实验。

下一阶段优先减少版本区间与 embedding 布局冲突导致的时间碎片，考虑时间前缀/分区摘要和紧凑数组节点，
并以建库+滚动查询+更新总成本检验是否有效；同时修复 PyTorch 环境、实现可训练预测条件编码与影响编码，
再接入经时间来源审计的真实文本/观测。不能把当前合成通过代替这些工作。
