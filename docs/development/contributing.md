# Contributing

## Development Setup

```bash
# Clone and install all development dependencies (CPU backend)
git clone https://github.com/avitai/DiffAV.git
cd DiffAV
./setup.sh --backend cpu

# Install pre-commit hooks
uv run pre-commit install
```

## Running Tests

```bash
# Full test suite with coverage
uv run pytest -vv --cov=src/ --cov-report=term-missing

# Specific test file
uv run pytest tests/data/test_wod_source.py -v

# Skip slow tests
uv run pytest -m "not slow"
```

## Code Quality

All code must pass the following checks before committing:

```bash
# Lint and format
uv run ruff check src/ --fix
uv run ruff format src/

# Type checking
uv run pyright src/

# Pre-commit (runs all hooks)
uv run pre-commit run --all-files
```

### Style Guidelines

- **Line length**: 100 characters
- **Formatter**: Ruff (not Black or yapf)
- **Type checker**: Pyright (not mypy)
- **Docstrings**: Google style
- **Imports**: isort ordering via Ruff

### Data Containers

Use frozen dataclasses for all domain types:

```python
@dataclass(frozen=True, slots=True, kw_only=True)
class MyType:
    """Description."""
    field: type
```

### Protocols

Use `typing.Protocol` with `@runtime_checkable` for dependency inversion:

```python
@runtime_checkable
class MyProtocol(Protocol):
    def method(self, x: InputType) -> OutputType: ...
```

## Pull Request Process

1. Create a feature branch from `main`
2. Write tests first (TDD), then implement
3. Ensure all tests pass and coverage >= 80%
4. Run `uv run pre-commit run --all-files`
5. Open a PR with a clear description of changes

## Project Structure

```
diffav/
├── src/diffav/       # Source code
│   ├── core/             # Domain types, protocols, config
│   ├── data/             # WOD source, parsers, converters
│   ├── models/           # Trajectory generation models
│   ├── alignment/        # DPO/RLHF alignment
│   ├── evaluation/       # Metrics and evaluation
│   ├── physics/          # Physics validation
│   ├── occupancy/        # Occupancy flow
│   ├── sensor/           # Sensor simulation (differentiable skeleton; not photorealistic)
│   └── api/              # Orchestration API
├── tests/                # Test suite (mirrors src/)
├── docs/                 # Documentation (MkDocs)
├── examples/             # Jupytext examples
└── scripts/              # Build and validation scripts
```
