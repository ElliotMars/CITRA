"""Train and evaluate CITRA without modifying legacy ReTaMTSF entry points."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import yaml

from data_provider.citra_weather import make_fixture
from exp.exp_CITRA import run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/citra.yaml"))
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mode", choices=["all", "ts_only", "semantic_topk", "full_asof_text", "shuffled_text", "citra_certified"], default="all")
    parser.add_argument("--availability-protocol", choices=["verified_availability", "issuance_asof_assumption"])
    parser.add_argument("--text-delay-hours", type=float)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--fixture", action="store_true", help="generated integration smoke, never real-data results")
    parser.add_argument("--allow-simulated", action="store_true", help="explicit latency scenario; excludes unknown availability")
    args = parser.parse_args()
    if args.checkpoint and args.mode == "all":
        parser.error("checkpoint evaluation requires one explicit mode")
    config = yaml.safe_load(args.config.read_text())
    if args.availability_protocol:
        config["availability_protocol"] = args.availability_protocol
    if args.text_delay_hours is not None:
        config["text_delay_hours"] = args.text_delay_hours
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if config["epochs"] <= 0 or config["batch_size"] <= 0 or config["stride"] <= 0:
        parser.error("epochs/batch_size/stride must be positive")
    if args.fixture and args.data:
        parser.error("use fixture or prepared real data")
    if args.fixture:
        fixture_root = Path(tempfile.mkdtemp(prefix="citra-learning-fixture-")) / "data"
        make_fixture(fixture_root, base_dim=config["model"]["base_dim"], seed=config["seed"])
        config["data_path"], config["allow_simulated"] = str(fixture_root), True
    elif args.data:
        config["data_path"] = str(args.data)
    if args.allow_simulated:
        config["allow_simulated"] = True
    output = args.output or Path("results/citra") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    results = run_experiment(config, output, args.device, args.mode, args.checkpoint)
    print(json.dumps(dict(output=str(output), is_fixture=results["is_fixture"], device=results["device"],
                          modes=list(results["models"])), indent=2))


if __name__ == "__main__":
    main()
