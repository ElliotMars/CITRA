"""Frozen, revision-pinned SentenceTransformer features; no labels are inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from time import perf_counter
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_provider.citra_weather import load_jsonl


PREPROCESSING = "remove-ascii-controls-except-whitespace;collapse-whitespace;retain-content-and-headers;tokenizer-truncation"


def preprocess(text):
    return re.sub(r"\s+", " ", re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)).strip()


def encode(records, output, model_name, revision, device="cpu", batch_size=64, max_length=256,
           local_model_path=None, backend="sentence_transformer",text_strategy="head"):
    from huggingface_hub import snapshot_download
    output = Path(output)
    if output.exists():
        raise FileExistsError("preserving existing offline embeddings")
    if not revision or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("pin an actual 40-character model commit, not a moving branch")
    started = perf_counter()
    path = local_model_path or snapshot_download(model_name, revision=revision, token=False,
        cache_dir="/tmp/citra-hf-cache", allow_patterns=["*.json", "*.txt", "*.safetensors", "1_Pooling/*"])
    acquisition_seconds = perf_counter() - started
    load_start = perf_counter()
    if backend == "sentence_transformer":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(str(path), device=device, local_files_only=True, trust_remote_code=False)
        model.eval().requires_grad_(False)
        model.max_seq_length = max_length
    elif backend == "transformers":
        import torch
        from transformers import AutoModel, AutoTokenizer
        pool = json.loads((Path(path) / "1_Pooling/config.json").read_text())
        if not pool.get("pooling_mode_mean_tokens") or any(v for k, v in pool.items()
                if k.startswith("pooling_mode_") and k != "pooling_mode_mean_tokens"):
            raise ValueError("direct Transformers backend supports the pinned mean-token pooling model only")
        tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
        model = AutoModel.from_pretrained(str(path), local_files_only=True, trust_remote_code=False).to(device)
        model.eval().requires_grad_(False)
    else:
        raise ValueError("unknown frozen encoder backend")
    loading_seconds = perf_counter() - load_start
    preprocess_start = perf_counter()
    texts = [preprocess(row["text"]) for row in records]
    if text_strategy not in ("head","tail"):
        raise ValueError("text strategy must be head or tail")
    if text_strategy=="tail":
        tok=model.tokenizer if backend=="sentence_transformer" else tokenizer
        texts=[tok.decode(tok(t,add_special_tokens=False,truncation=False)["input_ids"][-(max_length-2):],
                          skip_special_tokens=True,clean_up_tokenization_spaces=False) for t in texts]
    preprocessing_seconds = perf_counter() - preprocess_start
    encoding_start = perf_counter()
    if backend == "sentence_transformer":
        vectors = model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                               normalize_embeddings=True, show_progress_bar=True)
    else:
        from torch.nn import functional as F
        chunks = []
        with torch.no_grad():
            for begin in range(0, len(texts), batch_size):
                tokens = tokenizer(texts[begin:begin+batch_size], padding=True, truncation=True,
                                   max_length=max_length, return_tensors="pt").to(device)
                hidden = model(**tokens).last_hidden_state
                weights = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1e-9)
                chunks.append(F.normalize(pooled, dim=-1).cpu().numpy())
                print(f"encoded {min(begin+batch_size,len(texts))}/{len(texts)}", flush=True)
        vectors = np.concatenate(chunks)
    encoding_seconds = perf_counter() - encoding_start
    if not np.isfinite(vectors).all() or vectors.ndim != 2 or len(vectors) != len(records):
        raise ValueError("invalid frozen text embeddings")
    output.mkdir(parents=True)
    np.savez_compressed(output / "embeddings.npz", vectors=vectors.astype(np.float32))
    rows = [dict(doc_id=r["doc_id"], version_id=r["version_id"], vector_index=i,
                 raw_sha256=r.get("sha256"), text_sha256=hashlib.sha256(r["text"].encode()).hexdigest(),
                 preprocessed_sha256=hashlib.sha256(t.encode()).hexdigest())
            for i, (r, t) in enumerate(zip(records, texts))]
    (output / "identities.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    model_hashes = {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(path).rglob("*")
                    if p.is_file() and p.suffix in (".safetensors", ".json", ".txt")}
    manifest = dict(encoder_name=model_name, revision=revision, preprocessing=PREPROCESSING, backend=backend,
                    text_strategy=text_strategy,
                    max_wordpieces=max_length, normalized=True, frozen=True, dimensions=vectors.shape[1],
                    records=len(records), encoding_seconds=encoding_seconds, device=device,
                    model_acquisition_seconds=acquisition_seconds, model_loading_seconds=loading_seconds,
                    preprocessing_seconds=preprocessing_seconds, model_files_sha256=model_hashes,
                    embeddings_sha256=hashlib.sha256((output / "embeddings.npz").read_bytes()).hexdigest(),
                    identities_sha256=hashlib.sha256((output / "identities.jsonl").read_bytes()).hexdigest(),
                    source_records_sha256=hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest(),
                    note="No forecasting labels, PCA fitting or test-based model selection.")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_features(directory, records):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, field in (("embeddings.npz", "embeddings_sha256"), ("identities.jsonl", "identities_sha256")):
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != manifest[field]:
            raise ValueError("offline text feature hash mismatch")
    identities = load_jsonl(directory / "identities.jsonl")
    vectors = np.load(directory / "embeddings.npz", allow_pickle=False)["vectors"]
    table = {(r["doc_id"], r["version_id"]): r for r in identities}
    if len(table) != len(identities) or len(vectors) != len(identities):
        raise ValueError("duplicate/inconsistent offline feature identities")
    result = []
    for row in records:
        feature = table[row["doc_id"], row["version_id"]]
        if feature["text_sha256"] != hashlib.sha256(row["text"].encode()).hexdigest():
            raise ValueError("text changed since offline embedding")
        result.append(dict(row, embedding=vectors[feature["vector_index"]].astype(np.float64).tolist()))
    return result, manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--documents", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--revision", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--local-model-path", type=Path)
    p.add_argument("--backend", choices=["sentence_transformer", "transformers"], default="sentence_transformer")
    p.add_argument("--text-strategy",choices=["head","tail"],default="head",help="fixed first/last token segment; no label-based selection")
    args = p.parse_args()
    print(json.dumps(encode(load_jsonl(args.documents), args.output, args.model, args.revision,
                            args.device, args.batch_size, args.max_length, args.local_model_path, args.backend,args.text_strategy), indent=2))


if __name__ == "__main__":
    main()
