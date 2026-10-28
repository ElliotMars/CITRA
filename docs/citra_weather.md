# AFD + ASOS 最小真实数据接口

此文档保留阶段 A 的接口与小样本记录。最新三年真实档案、显式可用性假设、冻结 MiniLM 与实际预测实验见 [真实数据就绪性报告](citra_realdata_readiness.md)；下文“小样本”“本轮没有评测 MiniLM”均为阶段 A 的历史状态。

## 来源与覆盖

- [IEM NWS Text Product Archive](https://mesonet3.agron.iastate.edu/info/datasets/) 保存 NWS 文本，历史库可能包含后补档案；覆盖质量随年份不同。
- [AFD retrieval API](https://mesonet.agron.iastate.edu/cgi-bin/afos/retrieve.py?help=) 支持按 PIL、UTC 起止时间下载 ZIP，最多 9999 产品；起点包含、终点排除。
- [ASOS API](https://mesonet3.agron.iastate.edu/cgi-bin/request/asos.py?help=) 返回观测，不等同于该观测首次可访问的时间。本实现显式请求 UTC、tmpf、sknt 和 routine reports。

本轮实际下载 2020-01-01T00Z 至 2020-01-03T00Z 的样本：

|站点/办公室|AFD 数|小时观测|有限数值格|来源元数据确认 WFO|严格历史可用性|
|---|---|---|---|---|---|
|DSM / DMX|12|48|96|是|未验证|
|ORD / LOT|16|48|96|是|未验证|
|IAH / HGX|12|48|96|是|未验证|

来源 URL、实际 retrieved_at、SHA256、站点坐标/名称/来源 WFO 均随文件保留在
`artifacts/citra_stage_a/weather_probe/`。这仅证明该两天的档案可获取，不能证明连续年份完整。
网络出现过 ConnectionReset/IncompleteRead；下载器有有限重试，ORD 经 IEM 官方镜像取得。

## 下载

先下载小范围验证，再依覆盖率决定实验年份。以下只是请求示例，不代表所有年份已经准备好：

```bash
python scripts/download_nws_afd.py --office DMX \
  --start 2020-01-01T00:00Z --end 2023-01-01T00:00Z --output /tmp/AFDDMX.zip
python scripts/download_asos.py --station DSM \
  --start 2020-01-01T00:00Z --end 2023-01-01T00:00Z --output /tmp/DSM.csv
```

对 ORD/IAH 分别使用 LOT/HGX。ASOS 支持 `--server mesonet3.agron.iastate.edu`；来源元数据将记录实际主机。
若达到 9999 条限制则拒绝宣布完整，需缩短区间后自行按来源版本合并再导入。
脚本不覆盖已有文件，不从 ZIP 解压到任意路径；原始 ZIP/CSV 及 hash sidecar 必须一起保存。

## 时间语义与证据

AFD 的 issued_at 来自 ZIP 文件名中的完整 UTC 时间，并与 WMO 日/时/分头、PIL 和办公室交叉核对。
实际 ZIP 名如 `AFDDMX_202001010547.txt`，同时支持带完整 product ID 的文件名。
doc_id 按办公室和原始发行时间组成；不同原文哈希为不同 version_id，相同内容去重。
更正文本的历史可用时刻不能仅由原始发行头恢复，需要可用性 ledger。

available_at 默认 None，availability_provenance=unknown。历史 ingested_at 默认 None；
当前下载的实际 retrieved_at 独立保存，不伪造过去的系统摄入时间。
event_start/event_end 没有可靠解析时保持 None，不把发行日当作事件日。
未来有效的预报只要先前已公开即可合法访问；MOS 扩展 join 也遵循这一规则。

可用性 ledger 为 JSONL，每行至少：

```json
{"sha256":"原始产品字节哈希","available_at":"2020-01-01T05:49:00Z","evidence_url":"可核实的历史来源证据"}
```

数值 ledger 使用该原始 CSV 行 `json.dumps(row,sort_keys=True)` 的 SHA256 为标识，格式相同。
程序验证哈希绑定、时区、时间顺序与证据字段存在；证据真实性仍需来源审计，不能由非空 URL 自动证明。
严格模式只使用被证明的 available_at，未知记录不升级为 verified。

若仅研究延迟情景，可使用 `--simulate-text-delay-minutes` 与 `--simulate-numeric-delay-minutes`；
它们产生 simulated 标记，训练/审计时还必须显式 `--allow-simulated`，不属于获证严格历史实验。

## 冻结文本特征与准备

优先导入已有 MiniLM/SentenceTransformer 或其他冻结基础编码输出：

```json
{"doc_id":"DMX:202001010547","version_id":"原始字节SHA256","embedding":[0.1,0.2,0.3]}
```

每条原文可用 `scripts.prepare_citra_weather.afd_records()` 读取后交给离线编码器；需保持 ID/hash 对应。
本轮没有下载或评测 MiniLM；测试中的固定特征来自 fixture。
显式 `--hash-features` 提供确定性的词袋哈希基线，它不是预训练语义 embedding。

```bash
python scripts/prepare_citra_weather.py --afd /tmp/AFDDMX.zip --asos /tmp/DSM.csv \
  --office DMX --station DSM --output dataset/citra_weather \
  --train-end 2021-01-01T00:00Z --validation-end 2022-01-01T00:00Z --test-end 2023-01-01T00:00Z \
  --embeddings /tmp/frozen_embeddings.jsonl --embedding-version minilm-frozen-v1 \
  --availability-ledger /tmp/afd_availability.jsonl --numeric-ledger /tmp/asos_availability.jsonl
python scripts/validate_citra_timestamps.py --data dataset/citra_weather
```

输出 `numeric.npz`、`documents.jsonl`、`manifest.json`。数据内容、文档、编码版本、时间划分和
训练 Scaler 指纹绑定 checkpoint；不匹配时拒绝直接加载。

## 数值窗口规则

温度 Fahrenheit、风速 knots；对 `[hour,hour+1)` 内有限观测取平均，结果标在 hour+1。
结果不早于小时结束和所含记录中最晚的 available_at；其中任一值的可用性未知，该格可用性仍未知。
数值秒级存储对可用时刻向上取整，防止微秒级晚到被截断到 cutoff；分割边界必须对齐 UTC 整小时。
缺失小时为 NaN；历史窗口仅向前继承更早且在 cutoff 已可用的值，最前缺失则丢弃窗口。
预测标签不填补。Scaler 仅拟合训练期内、训练截止前已可用的有限数值。

训练/验证/测试标签窗口不得跨分割边界。训练标签须在 train_end 前可用；选模标签须在 validation_end
前可用。验证预测 cutoff 不早于 train_end，测试预测 cutoff 不早于 validation_end，防止模型本身
利用尚未完成拟合/选模时才得到的标签。严格测试比较的是这个时间协议下的预测。

当前三个地区的归档没有逐条历史可用性证据；实际严格检查按预期退出 2，结果保存在
`weather_probe/strict_readiness.json`。不得据此运行并宣称严格真实天气收益。
最小补充是审计历史接收/发布时间 ledger，或开展有真实接收时间记录的前瞻采集。

`asof_forecast_covariates()` 已提供 MOS 预报的合法发行时间/有效时间 join 接口；尚未实现 MOS 下载、
模型输入集成或强数值天气预报对照，不能声称文本有其之外的补充收益。
