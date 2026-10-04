"""Export the canonical race-car YAML examples for the web backend."""

import argparse
import json
from pathlib import Path

import yaml

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "race-cars"


def load_example_yaml(filename: str) -> dict:
    """Decode one YAML document from the race-car examples directory."""
    return yaml.safe_load((EXAMPLES / filename).read_text(encoding="utf-8"))


def export_examples() -> list[dict]:
    """Resolve catalog file references to transport-neutral input mappings.

    Each example takes its display name from its geometry document.
    """
    catalog = json.loads((EXAMPLES / "catalog.json").read_text(encoding="utf-8"))
    examples = []
    for entry in catalog:
        geometry = load_example_yaml(entry["geometry"])
        examples.append(
            {
                "key": entry["key"],
                "name": geometry["name"],
                **entry,
                "geometry": geometry,
                "sweep": load_example_yaml(entry["sweep"]),
            }
        )
    return examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(export_examples(), indent=2) + "\n", encoding="utf-8"
    )
