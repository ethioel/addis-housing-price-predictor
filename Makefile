# ============================================
# Makefile for addis housing price predictor
# ============================================

.PHONY: help install generate clean test coverage format lint notebook build publish

help:
	@echo "📋 Available commands:"
	@echo ""
	@echo "  make install      - Install dependencies"
	@echo "  make generate     - Generate housing dataset"
	@echo "  make clean        - Clean temporary files"
	@echo "  make test         - Run tests"
	@echo "  make coverage     - Run tests with coverage"
	@echo "  make format       - Format code with black"
	@echo "  make lint         - Lint code with flake8"
	@echo "  make notebook     - Launch Jupyter notebook"
	@echo "  make build        - Build package"
	@echo "  make publish      - Publish to PyPI"

install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt
	pip install -e .

generate:
	@echo "🏠 Generating housing dataset..."
	python src/data_generator.py

clean:
	@echo "🧹 Cleaning temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	find . -type d -name ".mypy_cache" -delete
	find . -type d -name ".ruff_cache" -delete
	rm -rf outputs/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/

test:
	@echo "🧪 Running tests..."
	pytest tests/ -v

coverage:
	@echo "📊 Running tests with coverage..."
	pytest tests/ --cov=src --cov-report=html --cov-report=term

format:
	@echo "🎨 Formatting code..."
	black src/ tests/ notebooks/

lint:
	@echo "🔍 Linting code..."
	flake8 src/ tests/ --count --max-line-length=100 --statistics
	mypy src/ tests/ --ignore-missing-imports

notebook:
	@echo "📓 Launching Jupyter notebook..."
	jupyter notebook notebooks/

build:
	@echo "📦 Building package..."
	python -m build

publish:
	@echo "🚀 Publishing to PyPI..."
	python -m twine upload dist/*

all: clean install test
	@echo "✅ All done!"
