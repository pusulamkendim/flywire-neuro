"""Read-only summaries for recorded DN response libraries and plans."""

from __future__ import annotations

import json
from pathlib import Path


DATASET_DIR = Path(__file__).resolve().parent / "datasets"


def list_dataset_summaries() -> list[dict]:
    summaries = []
    for path in sorted(DATASET_DIR.glob("*.json")):
        data = json.loads(path.read_text())
        summaries.append({
            "id": data.get("id", path.stem),
            "schema_version": data.get("schema_version"),
            "status": data.get("status", "unknown"),
            "sensory_contract_id": data.get("sensory_contract_id"),
            "trial_count": data.get("trial_count"),
            "available_channels": data.get("available_channels", []),
            "source_log": data.get("provenance", {}).get("source_log"),
        })
    return summaries
