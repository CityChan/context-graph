"""Lossless, snapshot-local text encoding for DiscoveryWorld public observations."""
from collections import Counter
from copy import deepcopy
import json

PROFILES = ("full", "compact_v1")
FORMAT = "discoveryworld.compact.v1"
LEGEND = ("ui is the complete current observation. At paths in tables, columns give the "
          "field order for rows. A cell {\"text\":i} means texts[i], using zero-based indices. "
          "Text indices are not object UUIDs and apply only to this observation.")


def dumps(value, profile="full"):
    if profile not in PROFILES:
        raise ValueError("Unknown DiscoveryWorld observation profile")
    return json.dumps(value, ensure_ascii=False, **({"separators": (",", ":")} if profile == "compact_v1" else {}))


def _at(value, path):
    for key in path:
        value = value[key]
    return value


def encode(ui, profile="full"):
    if profile == "full":
        return ui
    if profile != "compact_v1":
        raise ValueError("Unknown DiscoveryWorld observation profile")
    # Only object lists become tables. Unknown/nested fields remain verbatim.
    paths = [[key] for key in ("inventoryObjects", "accessibleEnvironmentObjects") if key in ui]
    nearby = ui.get("nearbyObjects")
    nearby = nearby.get("objects", {}) if isinstance(nearby, dict) else {}
    if isinstance(nearby, dict):
        paths.extend(["nearbyObjects", "objects", direction] for direction in nearby)
    eligible = []
    counts = Counter()
    for path in paths:
        rows = _at(ui, path)
        if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
            continue
        columns = list(rows[0])
        if not columns or any(set(row) != set(columns) or any(isinstance(v, (dict, list)) for v in row.values()) for row in rows):
            continue
        eligible.append((path, columns, rows))
        counts.update(v for row in rows for v in row.values() if isinstance(v, str) and len(v) >= 24)
    texts = [value for value, count in counts.items() if count > 1]
    indices = {value: i for i, value in enumerate(texts)}
    result = deepcopy(ui)
    for path, columns, rows in eligible:
        _at(result, path[:-1])[path[-1]] = {
            "columns": columns,
            "rows": [[{"text": indices[v]} if isinstance(v, str) and v in indices else v
                      for v in (row[key] for key in columns)] for row in rows],
        }
    return {"format": FORMAT, "legend": LEGEND, "texts": texts,
            "tables": [path for path, _, _ in eligible], "ui": result}


def decode(encoded):
    """Reconstruct the exact public UI values without earlier observations."""
    if encoded.get("format") != FORMAT:
        raise ValueError("Expected a compact DiscoveryWorld observation")
    ui = deepcopy(encoded["ui"])
    for path in encoded["tables"]:
        table = _at(ui, path)
        rows = []
        for cells in table["rows"]:
            if len(cells) != len(table["columns"]):
                raise ValueError("Invalid table width")
            rows.append(dict(zip(table["columns"], [encoded["texts"][v["text"]] if isinstance(v, dict) else v for v in cells])))
        _at(ui, path[:-1])[path[-1]] = rows
    return ui
