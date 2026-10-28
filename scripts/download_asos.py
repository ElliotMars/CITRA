"""Download UTC ASOS observations and station metadata for a checked region."""
import argparse
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen
from citra_download import fetch, read_url


REGIONS = {"DSM": ("DMX", "IA_ASOS"), "ORD": ("LOT", "IL_ASOS"), "IAH": ("HGX", "TX_ASOS")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", choices=list(REGIONS), required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--server", choices=["mesonet.agron.iastate.edu", "mesonet2.agron.iastate.edu", "mesonet3.agron.iastate.edu"],
                        default="mesonet.agron.iastate.edu", help="official IEM mirror for transient connection failures")
    args = parser.parse_args()
    office, network = REGIONS[args.station]
    metadata_url = f"https://{args.server}/geojson/network/{network}.geojson"
    stations = json.loads(read_url(metadata_url, limit=5 * 1024 * 1024))
    matches = [f for f in stations["features"] if f.get("id") == args.station]
    if len(matches) != 1:
        raise ValueError("station identity not verified by source metadata")
    station = matches[0]
    source_wfo = station["properties"].get("wfo")
    if source_wfo is not None and source_wfo not in (office, "K" + office):
        raise ValueError("source station/office association differs from requested pairing")
    query = urlencode(dict(station=args.station, network=network, sts=args.start, ets=args.end,
                           data="tmpf,sknt", tz="UTC", format="onlycomma", missing="M", trace="empty",
                           latlon="no", elev="no", report_type=3))
    record = fetch(f"https://{args.server}/cgi-bin/request/asos.py?" + query, args.output,
                   dict(timestamp_timezone="UTC", station_id=args.station, office_id=office,
                        station_metadata_url=metadata_url, station_metadata=station,
                        office_mapping_verified=source_wfo is not None, observation_time_is_availability=False))
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
