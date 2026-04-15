.PHONY: help install dev-install test lint format clean docker-build docker-run docker-stop train dashboard

# Default target
help:
	@echo "CognitionSim - Available Commands"
	@echo "========================================"
	@echo "install         - Install production dependencies"
	@echo "dev-install     - Install development dependencies"
	@echo "test            - Run tests with coverage"
	@echo "lint            - Run code linting"
	@echo "format          - Format code with black"
	@echo "clean           - Remove build artifacts and cache"
	@echo "docker-build    - Build Docker image"
	@echo "docker-run      - Run with Docker Compose"
	@echo "docker-stop     - Stop Docker containers"
	@echo "train           - Run training locally"
	@echo "dashboard       - Start web dashboard"
	@echo "setup-env       - Create .env from template"

# Installation
install:
	pip install -r requirements_dashboard.txt
	pip install torch snntorch datasets transformers networkx scipy scikit-learn matplotlib nltk sympy
	python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"

dev-install: install
	pip install pytest pytest-cov flake8 black pylint mypy

# Testing & Quality
test:
	pytest tests/ -v --cov=. --cov-report=html --cov-report=term

lint:
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics --exclude=__pycache__,venv,env
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics --exclude=__pycache__,venv,env

format:
	black . --exclude="/(\.git|\.venv|venv|env|__pycache__|\.pytest_cache)/"

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.log" -delete
	rm -rf .pytest_cache .coverage htmlcov/ dist/ build/ *.egg-info

# Docker
docker-build:
	docker build -t cognitionsim-ai:latest .

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

docker-logs:
	docker-compose logs -f

# Training & Dashboard
train:
	python train_atlantean_hrm.py

train-quick:
	python train_atlantean_hrm.py --num_batches 10 --batch_size 2

dashboard:
	python app.py

# Environment Setup
setup-env:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo ".env file created. Please edit it with your configuration."; \
	else \
		echo ".env file already exists."; \
	fi

# Model Management
save-models:
	@mkdir -p models
	@cp -v *.pth models/ 2>/dev/null || echo "No model files to save"

load-models:
	@cp -v models/*.pth . 2>/dev/null || echo "No model files to load"

# Development
dev: setup-env
	@echo "Starting development environment..."
	python app.py

prod: setup-env
	@echo "Starting production environment..."
	FLASK_ENV=production python app.py

# CI/CD simulation
ci: lint test
	@echo "CI checks passed!"

# Deployment
deploy-docker: docker-build docker-run
	@echo "Deployed with Docker Compose"

# Documentation (if you add docs later)
docs:
	@echo "Documentation generation not yet implemented"

# Database/State Management
backup-state:
	@mkdir -p backups
	@tar -czf backups/state-backup-$$(date +%Y%m%d-%H%M%S).tar.gz dashboard_state/ *.pth training_metrics.json
	@echo "State backed up to backups/"

restore-state:
	@echo "List of available backups:"
	@ls -1 backups/*.tar.gz 2>/dev/null || echo "No backups found"
