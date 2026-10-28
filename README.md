# CITRA — Certified Influence-aware Temporal Retrieval and Aggregation

CITRA 是面向历史可用文本的研究原型：可训练预测查询、带符号文本影响、完整合法核求和，以及相对于固定 Full Text 模型的解析近似误差界。尚未建立真实预测收益或 ICDE 研究创新性。

新增 PyTorch 模型见 `models/CITRA.py`，训练入口为 `run_CITRA.py`。数值输入 `[B,L,C]`，多步输出 `[B,H,C_target]`；原 ReTaMTSF 模型和实验入口保留在下方。

## 安装与原型验证

使用独立 Python 环境；本轮实际验证 Python 3.10、NumPy 2.2.5、PyTorch 2.5.1 CPU / 2.4.1 CUDA。

```bash
python -m pip install -r requirements-citra.txt
python -m unittest discover -s tests -v
python run_CITRA.py --fixture --mode all --device cpu --output /tmp/citra-training-smoke
```

`--fixture` 只生成明确标记的合成集成数据，不是实际天气实验；结果不会报告真实 MSE/MAE。输出目录必须不存在，以免覆盖已有结果。`all` 分别训练 TS-only、Semantic Top-K、Full As-of，CITRA 与 Full As-of 共享同一冻结 checkpoint。

当前机器现有 CUDA PyTorch 有 `iJIT_NotifyEvent` 环境问题；已验证的隔离运行方式、实际日志及依赖说明见 [阶段记录](docs/citra_progress.md)。一般正常安装的 PyTorch 环境不需要该兼容步骤。

## 真实数据与训练

AFD/ASOS 下载、时间审计、离线 embedding 导入和按年份划分的准备命令见 [天气数据协议](docs/citra_weather.md)。下载到的发布时间/观测时间不会自动视为历史可用时间；严格模式需要来源证据，延迟模拟必须显式启用。

```bash
python run_CITRA.py --data dataset/citra_weather --mode all --device cpu --output /tmp/citra-weather-run
python run_CITRA.py --data dataset/citra_weather --mode citra_certified \
  --checkpoint /tmp/citra-weather-run/full_asof_text.pt --output /tmp/citra-certified-eval
```

配置见 `configs/citra.yaml`，可修改 lookback、horizon、目标列、embedding/impact 维度和误差预算。配置、Scaler、数据指纹、训练日志、checkpoint、在线/离线成本均保存到输出目录。

本轮实际训练产物位于 `artifacts/citra_stage_a/`；[结果摘要](docs/citra_stage_a_results.json) 和 [测试日志](docs/citra_stage_a_tests.txt) 可核查。三个地区仅完成真实档案覆盖/格式探测，没有获证的历史可用时间，因此没有真实天气 MSE/MAE。

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
