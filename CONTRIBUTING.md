# Contributing to TG-graph

Thank you for your interest in contributing to **TG-graph**! This project focuses on improving traditional ML systems for real-time fraud detection by enriching them with graph-structured features.

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## How to Contribute

### Reporting Bugs

1. **Check existing issues** — Search the [issue tracker](../../issues) to see if the bug has already been reported.
2. **Open a new issue** — If not, open a new issue using the **Bug Report** template. Include:
   - A clear and descriptive title
   - Steps to reproduce the behavior
   - Expected behavior vs actual behavior
   - Environment details (OS, Python version, package versions)
   - Any relevant logs or screenshots

### Proposing Features

1. **Open an issue** using the **Feature Request** template.
2. Describe the problem you are trying to solve.
3. Describe your proposed solution.
4. Discuss the feature with maintainers before implementing.

## Pull Request Process

1. **Fork** the repository to your GitHub account.
2. **Create a branch** from `main` with a descriptive name:
   ```bash
   git checkout -b feat/add-graph-attention-layer
   ```
3. **Make your changes** following the [Code Style](#code-style) guidelines.
4. **Add tests** for any new functionality.
5. **Update documentation** if your changes affect the public API.
6. **Commit** with clear, descriptive commit messages following [Conventional Commits](https://www.conventionalcommits.org/):
   ```
   feat: add graph attention network for feature aggregation
   fix: correct temporal leakage in graph feature mapping
   docs: update README with GNN usage examples
   ```
7. **Push** your branch and open a Pull Request against `main`.
8. **Wait for review** — A maintainer will review your PR.

### PR Requirements

All pull requests must pass the following automated checks:
- [ ] **Lint & format** — Ruff, Black
- [ ] **Type check** — mypy
- [ ] **Tests** — pytest with coverage
- [ ] **Dependency audit** — pip-audit

Additionally:
- At least **1 maintainer approval** is required.
- All review conversations must be resolved.
- The branch must be up to date with `main`.

## Code Style

### Python

- Follow [PEP 8](https://peps.python.org/pep-0008/).
- Use [Black](https://black.readthedocs.io/) for formatting (line length: 100).
- Use [Ruff](https://docs.astral.sh/ruff/) for linting.
- Use [mypy](https://mypy-lang.org/) for type checking.
- All public functions and classes must have docstrings (Google style).

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

| Prefix | Use |
|:---|:---|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `docs:` | Documentation only |
| `test:` | Adding or updating tests |
| `refactor:` | Code refactoring (no feature/fix) |
| `ci:` | CI/CD changes |
| `chore:` | Maintenance tasks |

## Testing

- Write tests for all new functionality.
- Use [pytest](https://docs.pytest.org/) as the test framework.
- Place tests in the `tests/` directory, mirroring the source structure.
- Run the full test suite before submitting a PR:
  ```bash
  pytest tests/ -v --cov=src
  ```
- Minimum code coverage target: **80%**.

## Development Setup

1. Clone the repository with submodules:
   ```bash
   git clone --recurse-submodules https://github.com/yourusername/TG-graph.git
   cd TG-graph
   ```

2. Install dependencies:
   ```bash
   uv pip install -e ./gen-fraud-graph
   uv pip install -e ".[dev]"
   ```

3. Generate synthetic data:
   ```bash
   uv run python src/data/generate_data.py --scale 0.01 --output data/raw
   ```

4. Run a quick test:
   ```bash
   uv run python main.py --help
   ```

---

Thank you for contributing to **TG-graph**!