# Actual commands and reproduction inputs (2026-10-29)

Working directory: `/opt/data/private/ATM-ReTaMTSF`; branch `feature/citra-prototype`;
base commit `78af29edb21b8b93d795ca6528b1e027273bd40f`.
`registration.json` predates new control outcomes; `semantics/methods_frozen.json`
predates 2024 prediction evaluation. The latter contains every new checkpoint SHA.

Interpreter: `/root/miniconda3/envs/text4ts/bin/python`.
Timing/training commands used these process settings:

```bash
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export LD_PRELOAD=/tmp/citra-ittapi/libjitprofiling.so
python scripts/register_citra_final.py
python scripts/benchmark_citra_final.py
python scripts/run_citra_final_semantics.py train --device cuda:0
python scripts/benchmark_citra_final.py --archive-mode asof_prefix
```

The full-archive replay was supplemented by the as-of-only prefix benchmark to
avoid penalizing AVL with records that had not yet been published at a cutoff.
All warm samples and both archive policies remain in separate directories.
No GPU model training ran concurrently with either system timing pass. A source
data collection process briefly overlapped the second pass; this was not an
exclusive-machine benchmark. Method order was rotated and raw maxima retained.

The newly registered independent year was collected without inspecting its
forecast metrics. It retains the same availability assumptions, not verified
historical arrival evidence:

```bash
python scripts/download_nws_afd.py --office DMX --server mesonet3.agron.iastate.edu \
  --start 2024-01-01T00:00Z --end 2025-01-01T00:00Z \
  --output dataset/citra_real_dsm/raw/AFDDMX_2024.zip
python scripts/download_asos.py --station DSM --server mesonet3.agron.iastate.edu \
  --start 2024-01-01T00:00Z --end 2025-01-01T00:00Z \
  --output dataset/citra_real_dsm/raw/DSM_2024.csv
python scripts/build_citra_multiyear.py collect --raw dataset/citra_real_dsm/raw \
  --years 2020 2021 2022 2023 2024 --protocol issuance_asof_assumption \
  --numeric-delay-hours 1 --quarantine-numeric-duplicates \
  --output dataset/citra_real_dsm/collection_2024
```

`documents_2024.jsonl` is the suffix of `collection_2024/documents.jsonl`
after an exact equality check against the 8,766 previous records from
`collection_2023/documents.jsonl`. The old documents were not re-encoded.

```bash
PYTHONPATH=/tmp/citra-textdeps:/tmp/citra-encoder-minimal \
python scripts/encode_citra_text.py --documents dataset/citra_real_dsm/documents_2024.jsonl \
  --output dataset/citra_real_dsm/minilm_2024 \
  --revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 \
  --max-length 256 --device cuda:0 \
  --local-model-path /tmp/citra-hf-cache/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41
python scripts/append_citra_features.py \
  --old-records dataset/citra_real_dsm/collection_2023/documents.jsonl \
  --new-records dataset/citra_real_dsm/documents_2024.jsonl \
  --old-features dataset/citra_real_dsm/minilm_extended \
  --new-features dataset/citra_real_dsm/minilm_2024 \
  --output dataset/citra_real_dsm/minilm_extended_2024
python scripts/build_citra_multiyear.py prepare \
  --collection dataset/citra_real_dsm/collection_2024 \
  --features dataset/citra_real_dsm/minilm_extended_2024 \
  --output dataset/citra_real_dsm/prepared_2024
python scripts/run_citra_final_semantics.py evaluate --device cuda:0 --include-new
python scripts/run_citra_final_semantics.py analyze
python scripts/benchmark_citra_scaling.py prepare
python scripts/benchmark_citra_scaling.py all
PYTHONPATH=/tmp/citra-diagnosis-plots python scripts/report_citra_final.py
python scripts/verify_citra_final.py
python -m unittest discover -s tests -v
```

Scaling timing ran after model training and prediction evaluation finished.
Each N/scenario runs in its own process; Gaussian source moments use only
2020 frozen document representations, and all samples are marked synthetic.

Artifacts and original data were not overwritten. Scripts that create datasets
or timing directories fail on existing output. For an independent rerun, copy
`configs/citra_final.yaml`, change its `output`, and set `CITRA_FINAL_CONFIG` to
that copy before registering. Use `report_citra_final.py --docs-out NEW_DIR` for
fresh reports. The final decision report contains a complete fresh-output recipe.
The existing inputs and checkpoints are required; their hashes are registered.

The official ITT compatibility library is only needed on this machine's broken
Torch installation, as documented in the historical progress report. No fake
symbols or numerical substitutions were introduced. Standard Torch environments
can omit `LD_PRELOAD`; optional text and plot dependencies are in the existing
requirements files. `dependencies.json` records the actual package versions.
