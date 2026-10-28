请继续开发当前 GitHub 仓库 `ElliotMars/CITRA`，基于最新提交开展下一阶段研究。

本轮的最高目标不是扩展模型复杂度，而是：**得到第一组真实、无已知未来信息泄漏、可复现的时间序列预测结果，并为后续索引优化建立公平的性能测量基础。**

请直接修改代码，保留现有 NumPy 认证模块、Temporal AVL、ReTaMTSF 旧代码和全部历史实验产物。不要强制覆盖、删除或推送用户已有修改。

### 任务 1：解决真实数据回测阻塞

首先检查 `data_provider/citra_weather.py`、`scripts/prepare_citra_weather.py` 和现有下载脚本。

实现两种显式区分的可用性协议：

- `verified_availability`：沿用已有的严格时间可用性验证逻辑。
- `issuance_asof_assumption`：使用原始产品发行时刻加上明确的延迟假设作为可见性阈值，默认不得将这种数据标记为 `verified`。

实现 0、1、6、24 小时的文本延迟敏感性实验，并为数值观测建立单独、明确的可用性假设。不得把历史回填档案的下载时间、事件时间或文件修改时间伪装成首次公开时间。

获取至少一个地区的连续多年 AFD 和 ASOS 数据。按年份分批下载，检查档案完整性、记录数、缺失率、重复文档、时间戳和数据哈希。

保存 provenance manifest，并输出每个数据划分中实际可用的预测窗口数量。

### 任务 2：真实文本特征

增加冻结 SentenceTransformer/MiniLM 文本编码接口，保留当前词袋哈希作为消融基线。

离线编码必须按 doc_id/version_id 保存，记录编码器名称、版本、预处理策略和文本哈希。不得用测试标签训练或选择文本编码表示。

### 任务 3：真实预测基线

使用现有 `models/CITRA.py` 和 `exp/exp_CITRA.py`。

至少实现并运行：

- TS-only
- Semantic Top-K
- Full As-of Text
- Shuffled Text Control

在同一时间划分下比较 MSE/MAE，先使用轻量模型，随后根据结果加入一个较强的数值骨干。

训练、验证、测试严格按时间划分。所有标准化参数只由训练期确定。

至少报告三种随机种子，不能仅展示最优一次。

输出文本可用性延迟变化对模型表现的影响。

如果真实数据无法获取或不满足协议，必须明确报告失败原因；不允许将 fixture 结果标记为真实预测结果。

### 任务 4：修复性能评价口径

目前 `ts_only` 推理路径不应承担无关的文本索引构建或 Query Encoder 计算。

请修改评估接口，使不同方法只统计真实需要的在线工作，并分开报告：

- Query Encoding
- Exact / Approximate Aggregation
- CPU/GPU Data Transfer
- Forecast Inference
- Index Construction
- Index Maintenance
- End-to-End Latency

保证 Full Text 与 CITRA 共享相同的冻结模型参数。

对真实查询 embedding 分布，记录节点半径、局部上界、实际扫描比例和近似是否退化到精确计算。

### 任务 5：研究判断

在完成上述真实数据实验后，先不要自动实现复杂 Hybrid Index。

请根据实际结果判断：

1. Full Text 是否真正优于 TS-only？
2. 文本打乱后是否失去收益？
3. 认证式近似是否能够避免扫描全部文档？
4. 当前瓶颈主要来自时间混合、空间划分、误差界过松还是 Python 执行开销？
5. 新 Temporal-Spatial Hybrid Index 是否有足够证据支持开发？

最后提交已修改文件、实际测试命令、真实结果、失败项和研究结论。更新 `docs/citra_progress.md`，新增 `docs/citra_realdata_readiness.md`。

不要虚构 ICDE 创新或系统加速。请直接执行可运行的部分，并明确报告尚未获得的证据。

现在先检查最新仓库工作树，完成任务 1–3 的最小真实实验，再修正任务 4 的计时协议。