# Test directory for CognitionSim

This directory contains unit tests and integration tests for the project.

## Running Tests

### Local Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_config.py

# Run with verbose output
pytest -v
```

### Using Make

```bash
# Run tests with coverage
make test

# Run linting
make lint

# Run formatting
make format
```

### In Docker

```bash
docker-compose run cognitionsim pytest tests/
```

## Test Structure

```
tests/
├── __init__.py           # Test package initialization
├── test_config.py        # Configuration tests
├── test_app.py          # Flask app tests (to be added)
├── test_atlantean_hrm.py # Model tests (to be added)
└── conftest.py          # Pytest fixtures (to be added)
```

## Writing Tests

Example test:

```python
import pytest
from config import Config

def test_config_defaults():
    """Test default configuration values"""
    config = Config()
    assert config.FIELD_SIZE == 100
    assert config.DEVICE == 'cpu'

@pytest.fixture
def app():
    """Create Flask app for testing"""
    from app import app
    app.config['TESTING'] = True
    return app

def test_health_endpoint(app):
    """Test health check endpoint"""
    with app.test_client() as client:
        response = client.get('/health')
        assert response.status_code == 200
```

## Coverage Reports

After running tests with coverage:

```bash
# View HTML report
open htmlcov/index.html

# View terminal report
pytest --cov=. --cov-report=term
```

## CI/CD Integration

Tests run automatically on:
- Every push to main/develop
- Every pull request
- Manual workflow dispatch

See `.github/workflows/ci-cd.yml` for details.
