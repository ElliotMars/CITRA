# 本轮实际运行产物

- `cpu/`、`gpu/`：三个模式各 3 epoch 的真实训练日志、选定 checkpoint、配置、完整推理成本及证书检查。
- `fixture/`：这些训练实际使用的可复现合成输入；不是实际天气数据。
- `weather_probe/`：三个地区两天的真实下载样本和来源元数据，只用于格式/覆盖/时间审计，未用于模型训练。
- `sha256.json`：上述产物的文件哈希（不含本 README）。

所有训练结果都标记 is_fixture=true，未报告真实 MSE/MAE。在线模型为同一冻结参数下的 Full Text 与
CITRA；epsilon=0/0.01/0.1 的测试中均精确扫描全部合法文档，没有展示系统扫描节省。

CPU checkpoint 的复用示例（使用可正常导入的 PyTorch）：

```bash
python run_CITRA.py --data artifacts/citra_stage_a/fixture --allow-simulated \
  --mode citra_certified --checkpoint artifacts/citra_stage_a/cpu/full_asof_text.pt \
  --device cpu --output /tmp/citra-saved-checkpoint-evaluation
```

原始 resolved_config 和日志中的 /tmp 路径保留实际执行记录，没有改写为未实际执行的命令。
可重定位 fixture 后显式传入新路径；内容指纹保持一致。输出目录须不存在。
