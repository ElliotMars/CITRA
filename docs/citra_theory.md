# 可训练模型的认证适用条件（2026-10-28）

训练模型使用 X=[B,L,C]，数值骨干输出 [B,H,C_target]。q 由历史、目标 ID、horizon 编码后归一化；
z 为固定基础文本 embedding 的可训练投影，v 为基础 embedding 与固定文档 metadata 的小型有符号网络。
训练的 A_full 采用可微 RBF 全量合法求和，不经过 NumPy 缓存，不把 Top-K 当作 oracle。
底层神经网络如何产生 z/v 不影响冻结后的泰勒证明；前提是该次推理中表示和影响均固定、有限、版本一致。

G(A)=reshape(WA) 是唯一支持认证的修正结构。只对 A 的误差放大取 L=||W||_F>=||W||₂，
不需要给 query/impact encoder 或数值骨干假定 Lipschitz=1，因为它们在 Full 与 Fast 中完全相同。
反之，训练中更新 z/v 后旧节点的 V/M/S 不再合法，必须重新编码并构建；不能边训练边引用旧的固定摘要证书。
参数哈希与索引代次检测保证代码不会默默使用旧缓存。线性头的 Python 结构也受检查，
不能仅保留同名 weight 张量却换成平方/指数函数后仍引用线性 L。

F_TS 及查询编码必须是确定性的固定模型计算；不支持训练模式或随机推理层。
本文不把自动哈希当作对任意用户改写 Python 代码的形式化验证，证书针对代码定义的受支持模型。

epsilon 约束单样本、所有 horizon/target 的归一化输出 L2 差异，非标签 MSE。
如果将预测恢复到物理单位，均值相消，尺度对角变换给出 L_raw <= max(target_scaler_scale)*L；
不能直接把归一化界当作 Fahrenheit/knots 混合向量的原单位界。真实 MSE/MAE 单独逆 Scaler 后计算，
并提供 per-target 指标避免跨物理单位的平均值掩盖问题。

模型集成对照预先使用 atol=rtol=1e-10。当前是解析截断余项证书，仍没有包含所有 PyTorch/NumPy/
CUDA 运算舍入的形式化 IEEE 总误差界。实际 smoke 的全扫描零截断界可能伴随约 1e-16 的舍入差异，
记录这些差异并按明确容差检查，没有事后提高界或将偏差改为零。

## 既有核、矩与时态证明（以下保留）

## RBF 梯度与 Hessian

固定 q，delta=q-z，k=exp(-||delta||²/(2 sigma²))，sigma>0：

```
grad_z k = k*(q-z)/sigma²                            [p]
Hess_z k = k*((q-z)(q-z)^T/sigma⁴ - I/sigma²)         [p,p]
```

令 t=||q-z||²/sigma²。p>=2 时，正交方向的 Hessian 特征值绝对值为
f(t)/sigma²，f(t)=exp(-t/2)；径向绝对值为 g(t)/sigma²，g(t)=|t-1|exp(-t/2)。
p=1 时仅有径向特征值。f 单调递减；g 在 [0,1] 单调递减，在 [1,3] 递增，
在 [3,infinity) 递减。g(3)=2exp(-3/2)<1。因此全局谱范数界是 1/sigma²。

## 查询/节点相关的局部界

节点球 B(c,R) 包含所有文档，以及从 c 到每个文档的整个泰勒线段。
令 d_min=max(0,||q-c||-R)，d_max=||q-c||+R；
a=(d_min/sigma)²，b=(d_max/sigma)²。
在该允许距离区间上，Hessian 谱范数包络的上确界只需检查：

- p>=2 的 f(a)；
- g(a)、g(b)；
- 若 3∈[a,b]，检查 g(3)。

最大值除以 sigma² 得 H_P(q)。p=1 不包含正交特征值。
这是距离区间上的有效谱范数上界，而不是只在中心评价 Hessian；
节点球的实际点集可能更小，所以对节点本身仍可能保守。
实现用 log(|t-1|)-t/2 避免径向乘积的过早下溢，对距离端点做向外 nextafter，
并在正的极小界下保留正浮点下限，避免远查询返回伪零余项。
这些保护不是完整区间算术证明，数值范围仍见下文。

## 摘要与泰勒余项

```
c = mean z
V = sum v                                      [m]
M = sum (z-c)v^T                                [p,m]
S = sum ||z-c||² ||v||₂
A_taylor = k(q,c)V + M^T grad_z k(q,c)           [m]
B_taylor = H_P(q) S / 2
```

标量二阶泰勒余项满足 |k(q,z)-k(q,c)-grad_z k(q,c)^T(z-c)| <= H_P(q)||z-c||²/2。
乘以影响向量后，逐项三角不等式给出 L2 聚合界 B_taylor。
不能先对 v 求和再取范数；正负抵消不意味着余项为零。

另一个可证明的选择是返回该合法节点的零贡献：
||sum k(q,z)v||₂ <= k_max * T，
其中 T=sum||v||₂，k_max=exp(-d_min²/(2sigma²))。
当前取零近似的界与泰勒近似的界中较小者，同时采用对应的近似值；
没有按照观察到的 exact error 修正界。大偏移/大半径下该替代可能更有效，
但一般正负抵消仍使三角界很松，本轮没有解决所有影响抵消问题。

## 可局部合并的精确矩

内部节点仅保存子节点和固定大小摘要，不复制子树文档引用/版本列表。
对子树 1、2，n=n1+n2，c=(n1c1+n2c2)/n，V=V1+V2：

M = M1+M2+(c1-c)V1^T+(c2-c)V2^T。

为了正确维护 S，额外保存非负权重 w_d=||v_d||₂ 的总量 T、加权中心 u、
以及 Q=sum w_d||z_d-u||²。合并时：

```
T = T1+T2
u = u1 + (T2/T)*(u2-u1)
Q = Q1+Q2+(T1/T)*T2*||u1-u2||²
S = Q+T*||u-c||²
```

T=0 时所有影响为零，Q=S=0。以上由平方范数展开和加权中心的一阶矩为零得到。
这是并行方差合并形式，避免原始二阶矩减去大偏移的消减；仍有 float64 舍入误差。
半径合并为 max(R1+||c1-c||,R2+||c2-c||)，是保守包围半径，不一定等于实际最小最大半径。
它保证每个子球包含于父球，从而局部 Hessian 界适用于整个展开线段。

## 历史区间与版本唯一性证明

对通过 provenance 的版本，activation=max(available_at,valid_from,known ingested_at)。
修订优先级是 (available_at,append_sequence)，与独立 snapshot 的选择规则一致。
按 activation 扫描；在同一时刻先合并所有激活候选，再保留最高优先级。
候选只进入集合，不因 valid_to 到期而移除，因此最高优先级版本不会退回旧版本。
一个版本最多赢得一个区间，其终点是下一次更高优先级激活时刻，进一步被 valid_to 截断。
已过期后才摄入的高优先级版本仍是 tombstone，必须压制旧版本，虽然它没有可见区间。

由归纳法，在每个激活点之后，winner 恰是 snapshot 中已可访问、valid_from 已开始的最高优先级版本。
最终与 valid_to 相交得到相同合法集合；每个 doc_id 的可见区间互不重叠。
因此任意 cutoff 上不遗漏、不重复，也不悄悄换成最新版本。未知 availability 不生成区间。
追加乱序历史版本只重新扫描该 doc_id 的版本并修改其受影响区间，不重算其他文档。

AVL 节点维护 start_min/start_max/end_min/end_max（None 的 end 为正无穷）：
cutoff<start_min 或 cutoff>=end_max 时整节点不可见；
cutoff>=start_max 且 cutoff<end_min 时所有区间合法；
其余是混合节点，必须细分，叶节点逐记录过滤。
只有完全合法的节点能用 V/M/S 聚合近似，不能把非法版本的影响混入摘要答案。
树保持不可变节点，局部旋转/path copying 后发布新根；所有 cutoff 复用该根，
持有旧根的读视图保持旧数据状态。不是磁盘持久化，也没有并发事务协议。

## 查询与复杂度

合法 frontier 是当前可见文档的无重叠分区。粗节点细化时删除父贡献，再加入两个子贡献。
总聚合界 B_A=sum B_P，最大界优先细化直到 B_A<=epsilon；界按非负项重新求和，避免减法消减。
epsilon=0 沿时间过滤遍历后单次调用完整 oracle reduction，不返回虚假的近似零界。
固定根和查询时，精确扫描工作量随更小预算不减；实际误差和延迟不要求单调。

局部更新涉及 k 个变化区间，每次 AVL 插入/删除复制 O(log(N+1)) 节点，叶桶最多 leaf_size 个文档。
该 doc_id 的 r 个版本区间推导成本为 O(r log r)，不是对总库的线性重建。
数值摘要每节点 O(p*m+p+m)，记录/时间元数据仅叶级 O(N)。
完整驻留内存仍有 Python/NumPy 对象、文档向量和身份键成本，不能只报告 summary_bytes。
时间区间混杂时过滤仍可退化至 O(N)；词典序 embedding 布局不是最优空间聚类。
保守合并半径与版本碎片会降低局部界效率，本轮实测确认总体成本仍可能更高。

## 预测层绑定

G(A)=reshape(WA)，L=||W||_F >= ||W||₂。
预测层直接从本次 LinearParameters 冻结快照的实际 W 计算 L，不使用可重写的外部常数。
把输出 epsilon 转为聚合预算，再验证 L*B_A<=epsilon，必要时精确执行。
同一快照的 F_TS 在两输出之差中相消。返回输出形状 [1,C,H]，
认证范数包含该样本所有通道和预测步。

参数通过 set_parameters 原子替换并提升 revision，旧数组不可写。
PredictionResult 同时绑定 model_revision 与 store_generation；
assert_current 拒绝任何参数变化、文档/影响修订后的旧证书。
旧 NumPy 验证头没有 impact encoder 缓存，文档影响是显式版本数据；新增 PyTorch 部署缓存见上节，
不能修改编码器而不重新编码/维护索引。当前不是并发安全的模型训练系统。

## 数值和创新边界

证书是实数数学下的解析余项界，float64 验证不是形式化 IEEE-754 全误差认证。
主测试/系统对照预先使用 atol=1e-10、rtol=1e-10；
大偏移/异质尺度压力测试使用 1e-8*(1+||A_full||)，不依赖观察到的误差调整界。
oracle/最终分区采用逐分量 math.fsum；主要溢出、非有限摘要/范数/界会拒绝输入。
仍未证明全部矩合并、指数、范数和模型矩阵乘法的机器舍入误差上限。
极端下溢或 epsilon 接近机器精度时不能承诺总浮点差异严格 <=epsilon。
返回字段 certificate_scope 明确该范围；epsilon=0 使用精确完整求和路径。

证书不约束真实未来预测 MSE/MAE。泰勒定理、AVL、矩合并、半径包围和线性 Lipschitz 界
均是已有构件，不能作为论文原创贡献。潜在增量在于时间版本合法求和与输出预算的联合执行，
但本轮只有可验证原型，没有真实数据或总体系统优势证据。
