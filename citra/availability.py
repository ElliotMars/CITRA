"""Explicit replay assumptions, never certificates of first public availability."""
from collections import Counter
from datetime import timedelta
import math


VERIFIED = "verified_availability"
ASSUMED = "issuance_asof_assumption"


def issuance_records(records, delay_hours, parse_time, *, drop_ambiguous=True):
    if not math.isfinite(delay_hours) or delay_hours < 0:
        raise ValueError("text delay must be finite and nonnegative")
    counts = Counter(row["doc_id"] for row in records)
    ambiguous = {doc_id for doc_id, count in counts.items() if count > 1}
    result = []
    for raw in records:
        # Two distinct versions with the same original issuance do not supply
        # a known correction publication order. Exclude the whole group rather
        # than pretending the last archive copy was available at issuance.
        if drop_ambiguous and raw["doc_id"] in ambiguous:
            continue
        if not raw.get("issued_at"):
            raise ValueError("issuance assumption needs an actual product issuance")
        row = dict(raw)
        row.setdefault("original_available_at", row.get("available_at"))
        row.setdefault("original_availability_provenance", row.get("availability_provenance", "unknown"))
        row["available_at"] = (parse_time(row["issued_at"]) + timedelta(hours=delay_hours)).isoformat()
        row["availability_provenance"] = "simulated"
        row["availability_protocol"] = ASSUMED
        row["assumed_text_delay_hours"] = delay_hours
        result.append(row)
    return result, dict(ambiguous_doc_ids_excluded=sorted(ambiguous) if drop_ambiguous else [],
                        records_excluded=len(records) - len(result), delay_hours=delay_hours,
                        historical_first_availability_verified=False)
