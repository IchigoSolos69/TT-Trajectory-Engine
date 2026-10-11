"""Immutable original-recording resolution, shared by training-only tools."""
import json
from pathlib import Path
import re


def resolve_original_path(capture, captures, remember):
    """Resolve relocated bytes by immutable source SHA; never infer from names."""
    original=Path(capture["original_path"])
    expected=capture.get("original_sha256")
    if not isinstance(expected,str) or not re.fullmatch(r"[0-9a-f]{64}",expected):
        raise ValueError("Immutable original source SHA required")
    if original.is_file():
        remember(original,expected)
        return original.resolve()
    sidecar=Path(captures)/"source_locations.json"
    if not sidecar.is_file():raise FileNotFoundError(str(original))
    remember(sidecar)
    data=json.loads(sidecar.read_text(encoding="utf-8"))
    if data.get("schema_version")!=1 or not isinstance(data.get("sources"),dict):
        raise ValueError("Invalid versioned source-location registry")
    entry=data["sources"].get(expected)
    if entry is None:raise FileNotFoundError(str(original))
    paths=entry.get("paths") if isinstance(entry,dict) else None
    if not isinstance(paths,list) or any(not isinstance(p,str) or not Path(p).is_absolute() for p in paths):
        raise ValueError("Source-location paths must be explicit absolute paths")
    for candidate in map(Path,paths):
        if candidate.is_file():
            remember(candidate,expected)
            return candidate.resolve()
    raise FileNotFoundError(str(original))
