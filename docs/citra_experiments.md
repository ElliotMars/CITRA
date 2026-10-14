# 实验协议与实际运行

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
