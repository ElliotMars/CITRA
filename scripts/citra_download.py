"""Bounded downloads with exact source URLs, hashes, and actual retrieval times."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
import time
from http.client import IncompleteRead


def read_url(url, limit=100 * 1024 * 1024):
    request = Request(url, headers={"User-Agent": "CITRA-research-data-preparation/1.0"})
    for attempt in range(3):
        try:
            with urlopen(request, timeout=30) as response:
                payload = response.read(limit + 1)
            if len(payload) > limit:
                raise ValueError("response exceeds size limit; request smaller ranges")
            return payload
        except (URLError, ConnectionError, TimeoutError, IncompleteRead) as error:
            if isinstance(error, HTTPError) and error.code not in (429, 500, 502, 503, 504):
                raise
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def fetch(url, output, extra=None):
    output = Path(output)
    metadata = output.with_suffix(output.suffix + ".metadata.json")
    if output.exists() or metadata.exists():
        raise FileExistsError(f"preserving existing download: {output}")
    payload = read_url(url)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    record = dict(source_url=url, retrieved_at=datetime.now(timezone.utc).isoformat(),
                  sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload),
                  historical_ingested_at=None, availability_provenance="unknown")
    record.update(extra or {})
    metadata.write_text(json.dumps(record, indent=2) + "\n")
    return record
