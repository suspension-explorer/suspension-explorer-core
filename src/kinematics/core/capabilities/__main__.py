"""Generate a manifest using only the core package dependencies."""

import argparse
import json
from pathlib import Path

from kinematics.core.capabilities.manifest import CapabilityManifest, create_manifest


def main() -> None:
    """Write the deterministic manifest and optionally its JSON Schema."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Manifest JSON path")
    parser.add_argument("--schema-out", type=Path, help="Optional JSON Schema path")
    args = parser.parse_args()
    manifest = create_manifest()
    outputs = {args.out: manifest.model_dump_json(indent=2) + "\n"}
    if args.schema_out is not None:
        if args.schema_out.resolve() == args.out.resolve():
            parser.error("--out and --schema-out must be different paths")
        outputs[args.schema_out] = (
            json.dumps(CapabilityManifest.model_json_schema(), indent=2) + "\n"
        )
    for path, content in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
