"""Shared paths/configuration; all commands operate on this repository."""
from pathlib import Path
import json
import yaml

ROOT = Path(__file__).resolve().parents[2]


def config():
    return yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))


def findings():
    return yaml.safe_load((ROOT / "reports/findings.yaml").read_text(encoding="utf-8"))


def write_findings(value):
    (ROOT / "reports/findings.yaml").write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def write_json(path, value):
    path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
