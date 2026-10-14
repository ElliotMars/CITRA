# 模块与数据流（审查修复版）

原 ReTaMTSF 模型、数据路径、权重和训练入口未改变；新增 CITRA 保持独立。

```
Document + availability provenance + immutable embedding/impact
   -> TemporalDocumentStore._versions (append-only)
   -> 每 doc_id 激活扫描 -> 互斥可见区间
   -> TemporalCITRAIndex (leaf-oriented persistent AVL)
        写入/修订: 差分区间 + 局部 path copying + 合并 V/M/S/半径/时间极值
        cutoff 查询: 不可见节点剪枝 / 完全合法节点摘要 / 混合节点细分
   -> 聚合 QueryResult (aggregate, aggregate_bound, stats)
   -> FixedLinearForecaster 的冻结参数快照自动导出 L
   -> PredictionResult ([1,C,H], 输出界, 模型/文档版本)
```

- temporal.py：时间元数据、未知时间排除、独立全扫描 snapshot/oracle、区间推导及增量索引写入。
- index.py：只在叶节点保存 TemporalEntry；内部节点保存摘要、时间极值、平衡高度和分隔键。
  查询相关局部 Hessian 界、全局界 ablation、best-first、无认证 coarse 模式和内存估算。
- kernel.py：RBF/文档梯度、全球与球区间局部 Hessian 界、独立全求和 oracle。
- prediction.py：不可写模型数组、set_parameters、自动计算输出放大界、过期证书检测。
  仍是 NumPy 固定模型验证头，没有 PyTorch training loop 或文本 impact encoder。
- synthetic.py：带显式模拟来源的文档/修订/迟到生成，可用 indexed=False 建立无索引基线。
- run_citra_synthetic.py：12 组预测输出证书 CPU smoke。
- run_citra_benchmark.py：统一滚动 cutoff、四种系统方式及 global/local ablation、增量更新和内存诊断。
- tests/test_citra.py / test_citra_review.py：原数学矩阵及审查反例、乱序版本、局部更新、存储回归测试。
- scripts/audit_citra_data.py：本地 CSV 元数据审计。

**不再存在**单 (cutoff,generation) 快照树缓存；所有新/旧/乱序 cutoff 使用同一 maintained index。
写操作对受影响 doc_id 重推导区间，并仅修改变化区间对应的 AVL 路径。
索引节点不可变；持有旧根不会被新插入/修订改写。系统不主动保存所有旧根，也不提供磁盘持久化。

成本统计：nodes_visited 包含时间过滤与空间细化；metadata_records_checked 是逐记录时间检查，
documents_scanned 是查询阶段逐文档 kernel 计算，summary_kernel_evaluations 是中心核计算。
新查询的 index_document_reads/build_seconds 为零，因为向量索引读取移到写操作并通过
last_update_stats、total_update_document_reads、total_update_seconds 单独收费。
不能将这些查询期零构建开销表述为整个系统没有文档访问。
文档读取计数是摘要构建/叶桶重计算的逻辑记录访问，不计每次缓存数组或比较键的机器内存加载；
本原型尚无磁盘页 I/O 计数，因此不能据此宣称减少了真实数据库物理 I/O。

summary_bytes 仅是数值矩 payload；resident_bytes 是去重递归 Python/NumPy 估算，包含 ndarray
backing buffers、文档和元数据。benchmark 另报进程 RSS、tracemalloc 常驻/构建峰值，
以及 store+index 的联合估算；这些指标定义不同，不能混用。

依赖仍只有现有 NumPy 与标准库，未安装新包。Python 3.10.0 / NumPy 2.2.5 环境可运行。
PyTorch 仍因 iJIT_NotifyEvent 导入失败；没有改写全局环境或用虚假符号替代修复。
P1 的可训练多模态模型、真实数据、预测 MSE/MAE、并发事务与大规模持久化仍未完成。
