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
