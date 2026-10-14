# 审查修复交付记录

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
