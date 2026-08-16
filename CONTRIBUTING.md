# Contributing

Thank you for your interest in Suspension Explorer Core.

Contributions, bug reports, testing, benchmarks, documentation improvements, and technical discussion are all welcome.

## Licensing of contributions

Suspension Explorer Core is licensed under **AGPL-3.0-only** from version
0.6.0 onward. Alternative commercial licensing is available separately; see
the [License section of the README](README.md#license).

To keep that licensing model simple, accepted contributions must be available
for both the open-source project and any separately licensed commercial
versions.

By submitting a pull request and explicitly accepting the contributor terms
in the pull request template, you:

1. Confirm that you own the copyright in your contribution, or otherwise have
   the authority necessary to submit and assign it.
2. Assign the copyright in your contribution to **Nick McCleery**, as
   maintainer of Suspension Explorer Core.
3. Agree that the contribution may be used, modified, distributed,
   sublicensed, and relicensed as part of Suspension Explorer Core, including
   under AGPL-3.0-only and under separate commercial license terms.

Accepted contributions will remain available as part of the open-source
Suspension Explorer Core project under **AGPL-3.0-only**. Commercial
relicensing does not remove the contribution from the open-source project.

You retain the right to use any independently owned material that you created
before contributing it, provided that doing so does not conflict with rights
you have assigned in the contribution itself.

Do not submit code or other material that you do not have the right to
contribute. In particular, if your employer, university, client, or another
party may own the copyright in work you create, make sure you have the
necessary permission before submitting it.

A `Signed-off-by` line under the Developer Certificate of Origin (DCO) is not,
by itself, a substitute for accepting these contributor terms.

## Pull requests

Before opening a substantial pull request, consider opening an issue first to
discuss the proposed change. This is especially useful for new suspension
architectures, solver changes, public API changes, or larger refactors.

When submitting a pull request:

- Keep the change focused where practical.
- Include tests for new behaviour or bug fixes.
- Update documentation where behaviour or public APIs change.
- Make sure the relevant development checks pass.
- Accept the contributor terms shown in the pull request template.

Pull requests cannot be merged unless the contributor terms have been
explicitly accepted.

## Bug reports

Bug reports with a minimal reproducing geometry are particularly useful.

Where possible, please include:

- The Suspension Explorer Core version or commit where you observed the issue.
- A minimal reproducing geometry, preferably as YAML.
- The command, API call, or analysis that produces the problem.
- The behaviour you expected.
- The behaviour you observed.
- Any relevant traceback, warning, or solver diagnostics.

Please avoid including confidential or proprietary vehicle data in public
issues.

## Development

Common development commands are documented in the
[README](README.md#development).

```bash
just test
just check
just format
just spellcheck
```

Before submitting a pull request, please make sure the relevant tests pass and
that formatting, linting, and spelling checks are clean.

## Generated code

Some source files are generated rather than edited directly.

Generated analytical Jacobians live in
`src/kinematics/core/jacobians.py`. Edit their symbolic definitions in
`tools/generate_jacobians.py` and regenerate them with:

```bash
just generate-jacobians
```

Do not manually edit generated expressions unless the development
documentation explicitly says otherwise.

## License

The contributor terms above apply to contributions accepted into Suspension
Explorer Core.

For details of the outbound licensing of Suspension Explorer Core itself, see
the [License section of the README](README.md#license).
