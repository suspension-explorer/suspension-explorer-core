# Setup development environment.
setup:
    uv venv
    uv sync --all-extras --dev
    uv pip install -e .

# Install dependencies.
install:
    uv sync --all-extras --dev
    uv pip install -e .

# Install core dependencies exercised by CI; visualization remains optional.
install-ci:
    uv sync --dev
    uv pip install -e .

# Clean.
clean:
    rm -rf .venv
    rm -rf .pytest_cache
    rm -rf .ruff_cache
    rm -rf dist
    rm -rf build
    rm -rf *.egg-info
    find . -type d -name __pycache__ -exec rm -rf {} +

# Testing.
test:
    uv run pytest tests/ --cov=. --cov-report=term --durations=0

# Run performance benchmarks (deselected from the default suite).
bench:
    uv run pytest tests/benchmarks -m benchmark --benchmark-only

# Regenerate e2e test reference files after geometry.yaml changes.
regen-refs:
    @echo "Regenerating e2e test reference files..."
    uv run kinematics sweep --geometry tests/data/geometry.yaml --sweep tests/data/sweep.yaml --out tests/data/e2e/output.csv
    uv run kinematics sweep --geometry tests/data/geometry.yaml --sweep tests/data/sweep.yaml --out tests/data/e2e/output.parquet
    @echo "✓ Reference files regenerated successfully"
    @echo "Run 'just test-e2e' to verify the new reference files work correctly"

# Run the manual visualization test to generate a suspension animation.
generate-animation-test:
    uv run pytest tests/manual/test_run_with_viz.py -m manual -s

# Directly generate visualization without the test.
generate-animation:
    uv run kinematics sweep --geometry=tests/data/geometry.yaml --sweep=tests/data/sweep.yaml --out=results.csv --animation-out=anim.gif

# Print CSE snippets for Jacobian functions (paste into src/kinematics/jacobians.py).
generate-jacobians:
    uv run python tools/generate_jacobians.py

# Type check.
type-check:
    uv run ty check .

# Linting.
lint:
    uv run ruff check .

# Lint and type check.
check:
    just lint
    just type-check

# Formatting.
format:
    uv run ruff format .


# Spell check source code and comments.
spellcheck:
    uv run codespell src/ tests/

# Spell check and fix issues automatically.
spellcheck-fix:
    uv run codespell --write-changes src/ tests/

# Spell check and fix issues interactively.
spellcheck-interactive:
    uv run codespell --write-changes --interactive 3 src/ tests/
