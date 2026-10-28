"""Frozen lexical features for explicit baseline/fixture use (no torch dependency)."""
import hashlib
import re
import numpy as np


def hash_text(text, dimensions=32):
    if dimensions <= 0:
        raise ValueError("embedding dimension must be positive")
    result = np.zeros(dimensions, dtype=np.float64)
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        digest = hashlib.sha256(word.encode()).digest()
        result[int.from_bytes(digest[:4], "big") % dimensions] += 1 if digest[4] & 1 else -1
    norm = np.linalg.norm(result)
    return result / norm if norm else result
