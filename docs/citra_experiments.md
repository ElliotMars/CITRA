# 可训练预测闭环实验（2026-10-28）

本轮按 prompt.md 452 行后的“先完成第一阶段”顺序执行，完成可训练模型/最小真实数据接口的可复核代码。
第一阶段的模型实验使用明确的合成 fixture；真实档案由于历史可用性缺证，没有运行获证真实天气训练。
未提前堆叠未经验证的 Hybrid 或大规模 Accuracy–Latency 结果。

## 实际训练与复用命令

```bash
PYTHONPATH=/tmp/citra-torch-cpu /root/miniconda3/envs/text4ts/bin/python run_CITRA.py \
  --fixture --mode all --epochs 3 --device cpu --output /tmp/citra-stage-a-cpu-final
LD_PRELOAD=/tmp/citra-ittapi/libjitprofiling.so /root/miniconda3/envs/text4ts/bin/python run_CITRA.py \
  --fixture --mode all --epochs 3 --device cuda:0 --output /tmp/citra-stage-a-gpu-final
PYTHONPATH=/tmp/citra-torch-cpu /root/miniconda3/envs/text4ts/bin/python -m unittest discover -s tests -v
```

以上 /tmp 目录是实际运行路径，已保存产物到 `artifacts/citra_stage_a/`。新运行应选择不存在的输出目录。
标准可导入 PyTorch 环境直接使用 python，不需要本机故障兼容前缀。

TS-only、Top-K、Full As-of 采用相同种子、数值骨干、时间划分、训练 3 epoch、batch size=16、
AdamW 配置和训练顺序。TS-only 不使用文本组件；Top-K 有自己的同容量文本模型，按合法候选训练。
Full As-of/CITRA 对每个 checkpoint 使用同一冻结参数、表示和影响，不分别训练两个模型后比较误差界。
验证标签只用于选模，且必须在 validation_end 前可用，测试 cutoff 不早于该时间。

CPU/GPU 每个模式分别评估 18 个测试窗口；Full 与 CITRA 对照 epsilon=0/0.01/0.1，
每设备 54 次证书检查。全部在声明数值容差内通过；fixture 不报告真实 MSE/MAE 或文本贡献。
完整训练日志、最佳 checkpoint、版本指纹、各阶段成本见产物及 `citra_stage_a_results.json`。

|设备|Full As-of P50/P95 ms|AVL epsilon=0.1 P50/P95 ms|Hybrid|
|---|---|---|---|
|CPU|1.558 / 1.701|1.829 / 1.920|未实现/未测|
|A100 CUDA|3.075 / 11.580|3.157 / 11.580|未实现/未测|

这些是 18 个不同窗口的一次推理诊断，不是统计显著的速度结论。所有 epsilon 下累计扫描 819 个
合法文档（即全部合法文本），所以本轮没有展示近似查询节省；CUDA 的哈希/CPU 索引/传输和同步成本也不可省略。
表中 Full As-of 是冻结模型下的完整历史过滤求和，含 query encoding 与预测；它不是上轮仅 kernel 的计时。
Hybrid 列明确缺失，未用零延迟或外推数字代替实际执行。

真实源样本覆盖与历史可用性审计详见 `citra_weather.md`：三个站各 48 小时，所有历史可用性未知，
严格校验按预期拒绝，不能写作真实训练成功。下一阶段首先获得可用性 ledger/前瞻接收记录，
再做按年份划分的真实训练；同时以训练 encoder 的近距查询推进 Hybrid 和成本感知细化。

## 上阶段 NumPy 系统对照（保留，未因本轮训练而重算）

## 命令

```bash
PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/text4ts/bin/python -m unittest discover -s tests -v
/root/miniconda3/envs/text4ts/bin/python run_citra_synthetic.py --documents 256 --seed 0 --output docs/citra_review_smoke_results.json
/root/miniconda3/envs/text4ts/bin/python run_citra_benchmark.py --documents 256 4096 --cutoffs 0 10 30 --epsilons 0.01 0.1 1 --distances 0 50 --repeats 3 --updates 8 --output docs/citra_review_benchmark.json
python scripts/audit_citra_data.py
```

旧 citra_smoke_results.json 是第一版结果，保留；审查修复结果使用 review 前缀，protocol_version=2。
test_certified_bound 保留 1296 组参数：4 种子、维度 1/3/16、叶大小 1/4/16、
半径 1e-6/0.05/100、cutoff 0/10/30、epsilon 0/1e-5/0.05/100。
新增拒绝伪 Lipschitz、实际参数范数、证书失效、球区间 Hessian、远查询、异质影响/大偏移、
乱序激活/摄入/过期/tombstone、共享子树和每节点摘要对照测试。
单独核对聚合偏差和输出偏差，批量形状不作为批量证书证明。

## 统一系统对照

同一可复现输入和合法完整核求和语义下比较：

1. Exact Full Scan：无索引 store，每次 snapshot 扫全部版本，随后完整求和。
2. Temporal Filter + Exact：预先筛选快照后求和；每 cutoff 的准备成本单独计入总成本。
3. 无认证分层聚合：相同 maintained temporal tree 和合法性过滤，忽略预算，不虚称满足 epsilon。
4. CITRA：局部界 best-first；另报 global bound ablation，验证查询相关剪枝贡献。

各方法在相同滚动 cutoff、q、epsilon 上计时；第一查询与多次重复的 P50/P95 分别记录。
oracle 错误检查是离线工作，不用于停止条件、不修改界，不混入在线 query elapsed。
sum(query repeats) 给出本次滚动工作负载实际总成本，不仅展示单条宽松预算暖查询。

内存诊断独立启用 tracemalloc；执行计时另外在无追踪器的同一进程中重放相同种子输入。
无索引与 CITRA 两者都计算数据生成/建库成本，动态更新分别计时；总成本包含
初始写入 + 滚动查询 + 更新，不包含离线正确性核对。
重放进程的解释器/分配器已预热，因此建库时间不应描述为独立 fresh-process 冷启动结果。
每配置 3 次重复只支持诊断，不能提供统计显著性或普遍系统加速结论。

内存报告 numeric_summary_bytes、去重驻留估算、RSS 和 tracemalloc peak；
RSS 含解释器/分配器，tracemalloc 不捕获所有原生分配，驻留估算是对象图估算。
三者不能互相替代。新增额外加权矩和时间字段也有成本，减少重复引用不保证总内存一定低于旧版。
更新后的 verification 查询单独计时，不藏到索引写开销之外后声称更新和验证都免费。

## 当前证据与限制

12 组预测 smoke 和 36 组系统配置实际运行；全部预算认证在声明容差下通过。
4096 文档、q=0、epsilon=0.1 时 near 查询仍接近全扫描且比 full scan 慢；
远查询局部界减少细化，但混合版本叶节点仍需精确过滤/计算。
整体建库+滚动查询+更新仍高于全扫描，本轮没有建立总体性能优势。
详见 citra_progress.md 和完整 JSON，不能把局部查询优化替代端到端成本结论。

未运行 train/test 预测准确率实验，没有 MSE/MAE 或文本贡献结果。
后续必须训练可比 TS-only、Top-K、Full As-of 和 CITRA；
Full/CITRA 共享固定模型参数，所有 preprocessing/encoder 仅拟合训练期。
NWS AFD、ASOS、可选 MOS 的可用时间/摄入时间证据和真实数据导入仍待完成。
FAISS Top-K 如加入，只是有限候选近似，不是 full oracle。
