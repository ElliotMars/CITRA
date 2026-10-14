# CITRA 查询定义与证书 API

目标论文暂名 *Certified Temporal Evidence Aggregation for Text-Augmented Time Series Forecasting*。
本项目仍是研究原型，尚未证明真实预测收益、总体系统加速或 Research Track 创新增量。

逻辑查询为 q=(X_history,target,horizon,cutoff,epsilon)。当前索引接收历史信息生成的 q_z[p]、
UTC cutoff 和聚合误差预算；target/horizon 的可训练条件编码仍未实现。
固定有限向量 z_d、v_d 定义完整参考值：
A_full = sum_visible exp(-||q_z-z_d||²/(2 sigma²))*v_d。
必须完整合法求和，Semantic Top-K 不是 oracle。

文档必须在 cutoff 可访问，并采用当时的版本。available_at 未知或未核实的记录排除；
event_start/event_end 描述事件，不决定发布时间或可见性。未来事件可能已经提前预报。
known ingested_at 增加系统可访问性约束；未知摄入时间保留 None，不伪造。
有效区间是 [valid_from,valid_to)，修订采用 append-only。

**索引证书**：query_certified(query,cutoff,epsilon) 的 epsilon 是聚合 L2 误差预算。
QueryResult 只有 aggregate_bound，不包含预测 certified_bound，不接受任意 Lipschitz 数值。
解析余项满足 ||A_approx-A_full||₂ <= aggregate_bound <= epsilon。

**预测证书**：model.predict_certified(history,store,query,cutoff,epsilon) 的 epsilon 是单样本
预测输出 L2 预算。模型从实际固定权重快照计算可信 L，内部转换为聚合预算，
返回同一参数快照下的 prediction、certified_bound 和模型/文档库版本号。
参数或影响文档变化后必须重新查询；assert_current() 检测旧证书失效。
当前只支持单查询 history=[1,C,L]，输出 [1,C,H]；一般 predict 可用批量 [B,C,L]。
批量总 L2 预算不能直接复用单样本证书。

epsilon=0 精确遍历合法文档，空库返回零影响和零界。输出认证比较的是同一个模型、同一历史输入的
Full As-of 与近似预测，不保证未知未来标签的 MSE/MAE。float64 数值适用范围见理论文档。
