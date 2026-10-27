"""Validate native decision rows shared by the training reader and user kit.

Dataset ingestion, splitting and image checks live in openchoice.finetune.data.
Research-campaign generators and data acquisition are not part of this package.
"""
import math
from pathlib import PurePosixPath
import re

SCHEMA = "openchoice-decision-v1"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _finite(value):
    return type(value) in {int, float} and math.isfinite(value)


def validate_row(row):
    _require(isinstance(row, dict), "row must be an object")
    required = {"schema_version", "id", "group_id", "source_id", "split", "primitive", "family", "view", "context", "question", "target", "target_mask", "provenance"}
    _require(required <= row.keys() and not row.keys() - required - {"options", "assets"}, "missing or unknown row fields")
    _require(row["schema_version"] == SCHEMA, "unsupported row schema")
    for key in ("id", "group_id", "source_id", "family", "question"):
        _require(isinstance(row[key], str) and bool(row[key].strip()), f"invalid {key}")
    _require(isinstance(row["context"], str), "context must be text")
    _require(row["split"] in {"train", "dev", "probe"} and row["view"] == "base", "invalid split/view")
    _require(type(row["target_mask"]) is bool and isinstance(row["provenance"], dict), "invalid mask/provenance")
    primitive = row["primitive"]
    _require(primitive in {"choice", "score", "noul"}, "invalid primitive")
    options = row.get("options", [])
    _require(isinstance(options, list), "options must be list")
    ids = []
    for option in options:
        _require(isinstance(option, dict) and set(option) == {"id", "text"}, "invalid option fields")
        _require(all(isinstance(option[k], str) and option[k].strip() for k in ("id", "text")), "invalid option")
        ids.append(option["id"])
    _require(len(set(ids)) == len(ids), "duplicate option IDs")
    _require(not options if primitive == "noul" else 1 <= len(options) <= (10 if primitive == "score" else 26), "invalid option cardinality")
    target = row["target"]
    if not row["target_mask"]:
        _require(target is None, "masked targets must be null")
    else:
        _require(isinstance(target, dict), "missing target")
        if primitive == "noul":
            _require(set(target) == {"kind", "value"} and target["kind"] == "binary", "invalid binary target")
            value = target["value"]
            _require((type(value) is bool or _finite(value)) and 0 <= value <= 1, "invalid binary probability")
        elif target.get("kind") == "class":
            _require(set(target) == {"kind", "id"} and target["id"] in ids, "unknown target ID")
        else:
            _require(set(target) == {"kind", "values"} and target["kind"] == "distribution", "invalid distribution target")
            values = target["values"]
            _require(isinstance(values, dict) and set(values) == set(ids), "distribution support mismatch")
            _require(all(_finite(v) and 0 <= v <= 1 for v in values.values()) and math.isclose(sum(values.values()), 1, abs_tol=1e-7), "invalid distribution mass")
    assets = row.get("assets", [])
    _require(isinstance(assets, list), "assets must be list")
    for asset in assets:
        _require(isinstance(asset, dict) and set(asset) == {"path", "sha256", "media_type"}, "invalid asset fields")
        path = asset["path"]
        _require(isinstance(path, str) and path and not PurePosixPath(path).is_absolute() and all(p not in {"", ".", ".."} for p in path.split("/")) and "\\" not in path, "unsafe image path")
        _require(isinstance(asset["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", asset["sha256"]), "invalid asset hash")
        _require(asset["media_type"] in {"image/png", "image/jpeg"}, "invalid media type")
