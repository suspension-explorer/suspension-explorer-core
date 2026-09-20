# Capability manifest

The capability manifest is a versioned JSON contract for website and API
reference data. It describes built-in architectures, mechanisms, metrics, and
export columns.

## Generation

Generate the manifest and schema from a development checkout:

```bash
just manifest
# Write to a selected path:
just manifest /tmp/capabilities.json
```

The default outputs are `dist/capabilities.json` and
`dist/capabilities.json.schema.json`.

An installed package can generate the files outside a repository checkout:

```bash
python -m kinematics.core.capabilities --out capabilities.json --schema-out capabilities.schema.json
```

The Python API is `kinematics.core.capabilities.create_manifest()`. Generation
requires only core dependencies. CI generates the files from an installed wheel
and publishes them in the `core-capabilities` artifact.

## Contents and scope

The manifest contains:

- Package version, registered architectures and scopes, and supported setup shims.
- All valid combinations of finite mechanism selectors.
- Canonical metric metadata and per-configuration metric and output-point sets.
- Flat-export naming conventions.

`null` means an architecture does not expose a selector; `"none"` is an explicit
mechanism choice. Metric units preserve derivative operands, such as `mm/mm` or
`deg/mm`.

The catalog covers all built-in metric identities and finite mechanism
selections. It does not enumerate arbitrary hardpoints, numeric setup thicknesses,
custom derivatives, or every possible sweep target name. Setup shim support is reported
separately; pushrod shims require pushrod-rocker actuation. Declared metrics can
return empty values when a calculation is undefined or required physical inputs
are absent. Manifest generation leaves the CLI's CSV/Parquet format unchanged.

## Source declarations and validation

Generation enumerates the production selector enums and checks combinations
against the registered geometry schemas and actuation builder. It instantiates
accepted combinations with packaged reference hardpoints and reads the metric
and position declarations used by exports.

Generation does not run a sweep, require CLI extras, import test fixtures, or
infer compatibility from documentation. Reference coordinates provide valid
geometry for metadata discovery. They are not design recommendations and do not
guarantee validity for other hardpoints.

When adding an architecture or mechanism, update its production declarations
and, where needed, `src/kinematics/core/capabilities/reference.py` and
`src/kinematics/core/capabilities/reference_geometry.json`. Generation aborts if
an accepted reference combination fails to build. Coverage tests require every
registered architecture/scope and selector enum value to be represented, and
compare state declarations with actual state outputs.

Increment `schema_version` for a breaking manifest contract change.

## Website updates

In the website repository, ingest the released artifact with:

```bash
just reference-ingest /path/to/capabilities.json
```

Ingestion validates and saves the artifact, then generates the reference pages.
The website builds independently from its saved manifest.

Use separate feature branches in core and website. Merge and release core first,
then ingest the released artifact and merge the website update.
