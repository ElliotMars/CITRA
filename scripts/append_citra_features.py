"""Append separately encoded new-year text without changing existing vectors."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data_provider.citra_weather import load_jsonl
from scripts.encode_citra_text import load_features


def append(old_records,new_records,old_features,new_features,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    old,a=load_features(old_features,old_records);new,b=load_features(new_features,new_records)
    for field in ("encoder_name","revision","dimensions","max_wordpieces","normalized","model_files_sha256"):
        if a[field]!=b[field]:raise ValueError(f"foundation encoder differs: {field}")
    if a.get("text_strategy","head")!=b.get("text_strategy","head"):
        raise ValueError("text strategy differs")
    ids=load_jsonl(Path(old_features)/"identities.jsonl")+load_jsonl(Path(new_features)/"identities.jsonl")
    for i,row in enumerate(ids):row["vector_index"]=i
    np.savez_compressed(output/"embeddings.npz",vectors=np.asarray([r["embedding"] for r in old+new],dtype=np.float32))
    (output/"identities.jsonl").write_text("".join(json.dumps(r)+"\n" for r in ids))
    manifest=dict(a,records=len(ids),text_strategy=a.get("text_strategy","head"),
        encoding_seconds=a["encoding_seconds"]+b["encoding_seconds"],
        source_records_sha256=hashlib.sha256(json.dumps(old_records+new_records,sort_keys=True).encode()).hexdigest(),
        append_sources=[str(old_features),str(new_features)],original_vectors_preserved=True)
    for file,key in (("embeddings.npz","embeddings_sha256"),("identities.jsonl","identities_sha256")):
        manifest[key]=hashlib.sha256((output/file).read_bytes()).hexdigest()
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("old-records","new-records","old-features","new-features","output"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args();append(load_jsonl(a.old_records),load_jsonl(a.new_records),a.old_features,a.new_features,a.output)
