"""Download an immutable IEM NWS AFD zip. Issuance is not verified availability."""
import argparse
import json
from urllib.parse import urlencode
import zipfile
from citra_download import fetch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--office", required=True, choices=["DMX", "LOT", "HGX"])
    parser.add_argument("--start", required=True, help="UTC ISO inclusive, e.g. 2020-01-01T00:00Z")
    parser.add_argument("--end", required=True, help="UTC ISO exclusive")
    parser.add_argument("--output", required=True)
    parser.add_argument("--server", choices=["mesonet.agron.iastate.edu", "mesonet2.agron.iastate.edu", "mesonet3.agron.iastate.edu"],
                        default="mesonet.agron.iastate.edu")
    args = parser.parse_args()
    query = urlencode(dict(pil="AFD" + args.office, center="K" + args.office,
                           sdate=args.start, edate=args.end, fmt="zip", limit=9999, order="asc"))
    record = fetch(f"https://{args.server}/cgi-bin/afos/retrieve.py?" + query, args.output,
                   dict(office_id=args.office, start_inclusive=args.start, end_exclusive=args.end))
    with zipfile.ZipFile(args.output) as archive:
        count = len([n for n in archive.namelist() if not n.endswith("/")])
    record["product_files"] = count
    record["potentially_truncated"] = count >= 9999
    if count >= 9999:
        raise ValueError("AFD request may be truncated; split into smaller ranges")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
