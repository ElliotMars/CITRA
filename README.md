# CITRA — Certified Influence-aware Temporal Retrieval and Aggregation

最终诊断（2026-10-29）：**No-Go for Current CITRA**。
真实七天窗口平均约42篇；同冻结参数的NumPy Exact/AVL(ε=.1)端到端P50为0.864/2.707ms。
新2024年份真实文本MSE为0.308067，Metadata-only为0.307816，TS为0.309695；配对块Bootstrap未证明独立语义收益。
模型与AVL数学实现保留，未开发Hybrid。完整证据、合成规模的条件性优势与限制、83项测试和复现命令见
[最终系统报告](docs/citra_final_system_diagnosis.md)、[最终语义报告](docs/citra_final_semantic_diagnosis.md)、
[最终Go/No-Go决策](docs/citra_final_go_nogo.md)。

CITRA 是面向历史可用文本的研究原型：可训练预测查询、带符号文本影响、完整合法核求和，以及相对于固定 Full Text 模型的解析近似误差界。尚未建立真实预测收益或 ICDE 研究创新性。

新增 PyTorch 模型见 `models/CITRA.py`，训练入口为 `run_CITRA.py`。数值输入 `[B,L,C]`，多步输出 `[B,H,C_target]`；原 ReTaMTSF 模型和实验入口保留在下方。

此前阶段已完成 [预测退化诊断与受控消融](docs/citra_failure_diagnosis.md)：原始累计Sum存在尺度漂移，
新增Count Mean、公开时间窗口、固定TS残差与历史有界门控，均保留独立Exact Oracle和对应认证语义。
2021验证锁定后，2023保留年份的残差MSE由TS的0.301148降至0.298032；固定权重打乱文本后仍为0.298193，
尚不能确认可靠文本收益。75项测试和63组真实证书对照通过；本轮对Hybrid/复杂索引开发给出No-Go。

## 安装与原型验证

使用独立 Python 环境；本轮实际验证 Python 3.10、NumPy 2.2.5、PyTorch 2.5.1 CPU / 2.4.1 CUDA。

```bash
python -m pip install -r requirements-citra.txt
python -m unittest discover -s tests -v
python run_CITRA.py --fixture --mode all --device cpu --output /tmp/citra-training-smoke
```

`--fixture` 只生成明确标记的合成集成数据，不是实际天气实验；结果不会报告真实 MSE/MAE。输出目录必须不存在，以免覆盖已有结果。`all` 分别训练 TS-only、Semantic Top-K、Full As-of 和因果文本打乱对照，CITRA 与 Full As-of 共享同一冻结 checkpoint。

当前机器现有 CUDA PyTorch 有 `iJIT_NotifyEvent` 环境问题；已验证的隔离运行方式、实际日志及依赖说明见 [阶段记录](docs/citra_progress.md)。一般正常安装的 PyTorch 环境不需要该兼容步骤。

## 真实数据与训练

AFD/ASOS 下载、时间审计、离线 embedding 导入和按年份划分的准备命令见 [天气数据协议](docs/citra_weather.md)。下载到的发布时间/观测时间不会自动视为历史可用时间；严格模式需要来源证据，延迟模拟必须显式启用。

```bash
python run_CITRA.py --data dataset/citra_weather --mode all --device cpu --output /tmp/citra-weather-run
python run_CITRA.py --data dataset/citra_weather --mode citra_certified \
  --checkpoint /tmp/citra-weather-run/full_asof_text.pt --output /tmp/citra-certified-eval
```

配置见 `configs/citra.yaml`，可修改 lookback、horizon、目标列、embedding/impact 维度和误差预算。配置、Scaler、数据指纹、训练日志、checkpoint、在线/离线成本均保存到输出目录。

当前已完成 DSM/DMX 2020–2022 年真实档案实验：2020 训练、2021 验证、2022 测试，三种种子、四种文本延迟、冻结 MiniLM、四种预测方法，以及 DLinear/词袋哈希对照。真实数据采用明确的 `issuance_asof_assumption`，数值观测另设 +1 小时假设；这些假设不等于获证的历史首次公开时间。`verified_availability` 仍拒绝本批档案。

线性骨干零延迟的标准化测试 MSE（均值±种子标准差）：TS-only 0.3859±0.0063、Full Text 0.7017±0.0810。当前没有文本收益或系统加速证据，未实现 Hybrid Index。完整结果、失败项、下载/编码/复现实验命令见 [真实数据就绪性报告](docs/citra_realdata_readiness.md)；机器可读结果位于 `artifacts/citra_real_dsm/` 与 `artifacts/citra_real_hash/`。

```bash
python -m pip install -r requirements-citra-text.txt
python scripts/run_citra_real_experiments.py train --config configs/citra_real.yaml \
  --output /tmp/citra-real-rerun --device cuda:0
python scripts/run_citra_real_experiments.py profile_all --config configs/citra_real.yaml \
  --output /tmp/citra-real-rerun --profile-output /tmp/citra-real-profile --device cuda:0
```

以上命令需要先按报告准备本地真实数据；输出目录不会强制覆盖已有实验。旧阶段的 fixture 和真实小样本产物仍在 `artifacts/citra_stage_a/`，其历史结果不作为本轮真实预测指标。

---

# 原 ReTaMTSF：Retrieval-based Text-augmented Multivariate Time Series Forecasting

---

## Usage

### 1. Prepare Dataset

1. Download **Time-MMD** (https://github.com/AdityaLab/Time-MMD) and put it at `./dataset/`
2. Run preprocessing scripts for the specific dataset in `./pre_process/Time-MMD/`, including:

- `NumericalPackage.py`  
- `TextPackage.py`  
- `ChannelDescription_Pack.py`  
- `NewsDatabasePre_embed.py`  
- `Text_preembedding` (run twice: `--ForecastingPoint 0` for channel description, `--ForecastingPoint 1` for query texts)

---

### 2. Train and Evaluation

Run the training and evaluation script for the specific dataset:

```bash
bash ./run_XXX.sh  #(Replace XXX with the dataset name.)
```
