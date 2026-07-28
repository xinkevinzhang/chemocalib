# Contributing to ChemoCalib

Thank you for your interest in contributing to ChemoCalib!

## Development Setup

```bash
git clone <repo-url>
cd chemocalib
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
pip install -e ".[dev]"
```

## Code Style

- Follow PEP 8 with 100-character line limits.
- Use NumPyDoc-style docstrings for all public functions and classes.
- All new features must include unit tests.

## Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=chemocalib --cov-report=term

# Run a specific test file
pytest tests/test_core.py -v
```

## Pull Request Process

1. Create a feature branch from `main`.
2. Add tests for new functionality.
3. Ensure all tests pass locally.
4. Run `ruff check chemocalib/` to lint.
5. Submit a PR with a clear description of changes.

## Reporting Bugs

Please include:
- Python version (`python --version`)
- Operating system
- Minimal reproducible example
- Expected vs actual behavior

## Code of Conduct

This project follows the [Contributor Covenant](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).
