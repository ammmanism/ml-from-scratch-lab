.PHONY: test lint clean build check

test:
pytest tests/ -v

lint:
black src/ tests/
flake8 src/ tests/
mypy src/

clean:
find . -type d -name "__pycache__" -exec rm -rf {} +
rm -rf .pytest_cache
rm -rf .mypy_cache

build:
python -m build
