# CITRA 原型（审查修复版）

仅依赖 NumPy 和 Python 标准库，与原 ReTaMTSF 路径保持独立。

从仓库根目录运行：

```bash
PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/text4ts/bin/python -m unittest discover -s tests -v
/root/miniconda3/envs/text4ts/bin/python run_citra_synthetic.py --documents 256 --seed 0
/root/miniconda3/envs/text4ts/bin/python run_citra_benchmark.py --documents 256 4096 --cutoffs 0 10 30 --distances 0 50 --repeats 3
python scripts/audit_citra_data.py
```

在其他环境中用安装了 NumPy 的 Python 替代绝对路径即可，无需模型下载或 GPU。

```python
import numpy as np
from citra.synthetic import synthetic_store, EPOCH
from citra.prediction import FixedLinearForecaster
from datetime import timedelta

store = synthetic_store(128)  # 显式 simulated；内部启用 allow_simulated
cutoff = EPOCH + timedelta(days=30)
query = np.zeros(4)
aggregation = store.query_certified(query, cutoff, epsilon=0.1)
print(aggregation.aggregate_bound)  # 只认证聚合，没有 certified_bound 字段

model = FixedLinearForecaster(12, 2, 3, store.impact_dim)
result = model.predict_certified(np.zeros((1, 2, 12)), store, query, cutoff, epsilon=0.1)
result.assert_current()
print(result.prediction.shape, result.certified_bound)
```

**API 变更**：索引/文档库不再接受 lipschitz 参数，传入会抛出 TypeError。
预测层只从本次使用的固定参数快照计算 L。使用 set_parameters 修改权重后，或文档插入/修订
改变影响向量后，旧 PredictionResult.assert_current() 会拒绝旧证书；必须重新查询。
当前模型是固定 NumPy 参数验证头，set_parameters 支持显式参数更新，不是 trainable PyTorch 模型。

所有 cutoff 复用一个带可见区间的 AVL 索引，写操作局部维护摘要；不再每次重建快照树。
snapshot/aggregate_exact 保留独立全版本扫描，作为正确性 oracle。indexed=False 可建立无索引全扫描基线。
未知 available_at 默认排除；真实可用时间须保留来源证据，模拟时间必须显式标记并启用。
同 doc_id 的版本通过 append_revision 追加；事件日期不代替 available_at。

证书范围是解析截断误差界及明确 float64 容差，不是形式化 IEEE 舍入证明。
旧 docs/citra_smoke_results.json 保留为第一版历史结果；本轮文件使用 citra_review_*。
详细证明、局限与实测成本见 docs/citra_theory.md、citra_progress.md。
