"""Validated physical-world and sensory-contract configuration loader."""

from __future__ import annotations

import json
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parent
WORLD_DIR = BACKEND_DIR / "worlds"
CONTRACT_DIR = BACKEND_DIR / "sensory_contracts"


class WorldConfigError(ValueError):
    pass


def _safe_json(directory: Path, config_id: str) -> Path:
    if Path(config_id).name != config_id:
        raise WorldConfigError("Unsafe config id")
    path = (directory / f"{config_id}.json").resolve()
    if path.parent != directory.resolve():
        raise WorldConfigError("Config escapes its directory")
    return path


def load_sensory_contract(contract_id="sensory_contract_v1") -> dict:
    path = _safe_json(CONTRACT_DIR, contract_id)
    if not path.exists():
        raise FileNotFoundError(path)
    contract = json.loads(path.read_text())
    if contract.get("schema_version") != "flywire-sensory-contract-v1":
        raise WorldConfigError("Unsupported sensory contract")
    if not isinstance(contract.get("channels"), dict) or not contract["channels"]:
        raise WorldConfigError("Sensory contract has no channels")
    return contract


def load_world(world_id="microhabitat_v1", contract_id="sensory_contract_v1") -> dict:
    path = _safe_json(WORLD_DIR, world_id)
    if not path.exists():
        raise FileNotFoundError(path)
    world = json.loads(path.read_text())
    contract = load_sensory_contract(contract_id)
    if world.get("schema_version") != "flywire-world-v1":
        raise WorldConfigError("Unsupported world schema")
    if world.get("units") != "mm":
        raise WorldConfigError("World v1 must use millimetres")
    arena = world.get("arena", {})
    if not all(isinstance(arena.get(key), (int, float)) and arena[key] > 0
               for key in ("width_mm", "depth_mm")):
        raise WorldConfigError("World arena dimensions must be positive")

    entity_ids = set()
    known_channels = set(contract["channels"])
    for entity in world.get("entities", []):
        entity_id = entity.get("id")
        if not entity_id or entity_id in entity_ids:
            raise WorldConfigError(f"Missing or duplicate entity id: {entity_id}")
        entity_ids.add(entity_id)
        position = entity.get("position_mm")
        if not isinstance(position, list) or len(position) != 3:
            raise WorldConfigError(f"Entity {entity_id} needs position_mm [x,y,z]")
        unknown = set(entity.get("sensory", {})) - known_channels
        if unknown:
            raise WorldConfigError(f"Entity {entity_id} uses unknown channels: {sorted(unknown)}")
    world["sensory_contract_id"] = contract["id"]
    world["sensory_channel_count"] = len(contract["channels"])
    return world
