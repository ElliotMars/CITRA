# CITRA 最终Go/No-Go决策（2026-10-29）

**最终建议：No-Go for Current CITRA。停止当前论文主线，不开发Hybrid或新复杂索引。**

## 原始研究主张与已验证部分

原主张是在Point-in-Time文档可见性/版本规则下，用层次矩摘要和泰勒余项近似完整合法核和，
将误差传到同一冻结预测函数的输出，并同时获得有意义的预测收益和系统效率。
已验证：独立snapshot/oracle与版本区间语义、未知可用时间排除、合法窗口/计数归一、
固定参数的解析聚合余项、实际线性头/门控范数传播、参数/语料变更失效检查，以及真实可训练的预测路径。
这些是可复用的正确性/工程成果；不是形式化IEEE总舍入认证，也不自动构成ICDE方法创新。

## 两个维度独立决策

|维度|决策|实际证据|
|---|---|---|
|System|No-Go|真实七天P50=42篇、P95=44、Max=47；强Exact已亚毫秒；as-of前缀AVL(.1)仍更慢|
|Prediction|Evidence Insufficient / semantic claim No-Go|新2024真实文本不优于Metadata-only点估计；主指标配对CI跨0，部分气温指标更差；旧年份不稳定|
|论文路线|No-Go for Current CITRA|系统必要性和独立文本信息价值均未达到证据门槛|

System的同参数前缀库NumPy/AVL(.1)端到端P50=0.864/2.707ms，
纯向量化聚合只需0.035ms；没有可信真实工作负载支撑把当前42篇扩成数千/数万篇。
合成同质场景即使有交叉点，也不抵消真实查询更慢、构建/维护/内存额外成本或混合归档退化。

Prediction在2024的三种子均值：TS0.309695、真实0.308067、Metadata-only0.307816、
Content-only0.308539、Constant0.308377、Null0.308837、同权重打乱0.308240、重训打乱0.308369。
不可把相对TS约0.53%的总残差改善称为语义贡献；其7/14日块CI连相对TS的改善也未稳健排除0。
2023此前已看过，本轮不把它再称为独立测试。2024是锁定后的新年份，但仍只有同一地区，
首次历史公开时间没有获证，始终明确为issuance_asof_assumption。

## 值得保存的成果与停止边界

- Point-in-Time权威版本选择、可见区间推导、未知/模拟/verified的显式区分。
- 独立Exact Oracle、float64容差测试、解析余项与冻结模型输出预算的绑定。
- Temporal AVL的不可变节点/局部维护原型与完整负性能记录。
- Count Mean/七天窗口算子、固定TS残差和输入白名单归因控制。
- 连续真实原文/观测、SHA、冻结MiniLM身份绑定、所有种子/预测/配对日期与再现命令。
- 避免种子伪重复的日历块Bootstrap、严格区分旧诊断年份和新锁定验证。

停止当前主线不等于否认这些组件可在别处复用；本轮没有可信新应用场景，故不使用空泛Conditional Go延长项目。
不自动进入Hybrid、模型扩容、C++/CUDA重写或新数据结构开发。原ReTaMTSF路径和全部历史产物不改，不推送远端。

## 产物和复现

系统证据：[citra_final_system_diagnosis.md](citra_final_system_diagnosis.md)。
语义证据：[citra_final_semantic_diagnosis.md](citra_final_semantic_diagnosis.md)。
原始注册、环境/commit/checkpoint/source SHA、逐次性能、模型、预测、日期损失、Bootstrap、图、异常和测试在
[artifacts/citra_final_diagnosis](../artifacts/citra_final_diagnosis/)。完整协议为`configs/citra_final.yaml`。

本机Python3.10、NumPy2.2.5、Torch2.4.1/CUDA12.1/A100，Torch/NumPy CPU线程固定1。
因已有iJIT环境问题使用此前官方ITT兼容实现：
`LD_PRELOAD=/tmp/citra-ittapi/libjitprofiling.so`。正常Torch环境不需要该步骤。
绘图使用可选`requirements-citra-diagnosis.txt`，语义基础编码用`requirements-citra-text.txt`。
首次运行以下命令前，先按上轮天气报告及本轮commands文件准备被Git忽略的数据；
已有输出不会强制覆盖，重跑应复制协议并将output改到新目录：

```bash
python - <<'PY'
import yaml
from pathlib import Path
c = yaml.safe_load(Path('configs/citra_final.yaml').read_text())
c['output'] = '/tmp/citra-final-rerun'
Path('/tmp/citra-final-rerun.yaml').write_text(yaml.safe_dump(c,sort_keys=False))
PY
export CITRA_FINAL_CONFIG=/tmp/citra-final-rerun.yaml
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/register_citra_final.py
python scripts/benchmark_citra_final.py
python scripts/benchmark_citra_final.py --archive-mode asof_prefix
python scripts/run_citra_final_semantics.py train --device cuda:0
python scripts/run_citra_final_semantics.py evaluate --device cuda:0 --include-new
python scripts/run_citra_final_semantics.py analyze
python scripts/benchmark_citra_scaling.py prepare
python scripts/benchmark_citra_scaling.py all
python scripts/report_citra_final.py --docs-out /tmp/citra-final-reports
python scripts/verify_citra_final.py
python -m unittest discover -s tests -v
```

上面复制协议只更换产物位置，不改变冻结方法或种子；报告依赖原始年份数据和既有checkpoint。
新年份采集、单独编码/追加及实际带环境的执行命令见[commands.md](../artifacts/citra_final_diagnosis/commands.md)。
失败/异常在[failures.jsonl](../artifacts/citra_final_diagnosis/failures.jsonl)，没有删除失败种子或把合成曲线写成真实应用证据。
实际最终回归83项通过，其中新增8项覆盖非内容输入不可访问、直接月份清零、数值/合法集合不变、
缓存筛选与版本Oracle、种子不扩大天气样本量、块相关性/日期缺失及重复/跨年拒绝；
见[tests_final.txt](../artifacts/citra_final_diagnosis/tests_final.txt)。核心模型、原独立Oracle和AVL实现与起始commit保持一致，
未修改原ReTaMTSF路径；新增代码是诊断、控制、统计和报告脚本。
独立核对21份checkpoint、20个原始/元数据文件、72组预测、42个主指标配对区间、
3374次真实性能调用以及10个合成场景的500次查询通过，见[verification.json](../artifacts/citra_final_diagnosis/verification.json)。
训练时源码另有SHA匹配快照；后加的fresh-output配置入口不改变已登记的输入控制、统计方法或模型数学。
