# DevOps & Deployment Guide

## Overview

This guide covers the professional DevOps infrastructure for CognitionSim, including Docker containerization, CI/CD pipelines, and production deployment.

## 🐳 Docker Setup

### Quick Start with Docker

```bash
# Build the image
docker build -t cognitionsim-ai:latest .

# Run the container
docker run -p 5000:5000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/dashboard_state:/app/dashboard_state \
  cognitionsim-ai:latest
```

### Docker Compose (Recommended)

```bash
# Start the application
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the application
docker-compose down
```

### Environment Variables

Create a `.env` file from the template:

```bash
cp .env.example .env
# Edit .env with your configuration
```

Key environment variables:

- `FLASK_ENV`: `development`, `production`, or `testing`
- `SECRET_KEY`: Secret key for Flask sessions (change in production!)
- `HOST` / `PORT`: Server host and port
- `MODEL_ARTIFACTS_DIR`: Directory for model storage
- `FIELD_SIZE`: Neural field size (default: 100)
- `NUM_BATCHES`: Training batches (default: 100)
- `LOG_LEVEL`: `DEBUG`, `INFO`, `WARNING`, `ERROR`

## 📦 Model Artifact Management

### Directory Structure

```
models/                        # Persistent model storage
├── oscillator_weights.pth
├── atlantean_hrm_enhanced.pth
└── atlantean_hrm_weights.pth

dashboard_state/               # Training state
├── oscillator_weights.pth
├── metrics_history.pkl
└── system_state.pkl

logs/                          # Application logs
└── app.log
```

### Separating Model Storage

Models are stored separately from the application code for:

- **Version Control**: Don't commit large binary files to Git
- **Persistence**: Models survive container restarts
- **Sharing**: Multiple containers can share model artifacts
- **Backup**: Easy to backup/restore model states

### Backing Up Models

```bash
# Create backup
make backup-state

# Restore from backup
tar -xzf backups/state-backup-TIMESTAMP.tar.gz
```

## 🔄 CI/CD Pipeline

### GitHub Actions Workflows

Two automated workflows are included:

#### 1. CI/CD Pipeline (`.github/workflows/ci-cd.yml`)

Runs on every push and pull request:

- **Code Quality**: Linting with flake8, formatting with black
- **Testing**: Pytest with coverage reporting
- **Security**: Trivy vulnerability scanning
- **Docker Build**: Multi-platform image build (amd64, arm64)
- **Docker Push**: Automatic push to Docker Hub on releases
- **Artifacts**: Upload source and model artifacts

#### 2. Model Training Pipeline (`.github/workflows/model-training.yml`)

Scheduled and manual training:

- Runs weekly (Sunday 2 AM UTC) or on-demand
- Performs full model training
- Uploads trained models as artifacts
- Creates releases for scheduled runs

### Setting Up CI/CD

1. **Add GitHub Secrets**:
   ```
   DOCKER_USERNAME: Your Docker Hub username
   DOCKER_PASSWORD: Your Docker Hub token
   ```

2. **Enable Actions**:
   - Go to repository Settings → Actions → General
   - Enable "Allow all actions and reusable workflows"

3. **Trigger Workflows**:
   ```bash
   # Push triggers main CI/CD
   git push origin main
   
   # Manual training trigger
   gh workflow run model-training.yml
   ```

## 🚀 Deployment Options

### Local Development

```bash
# Using Makefile
make dev

# Or directly
python app.py
```

### Production with Docker Compose

```bash
# Set production environment
echo "FLASK_ENV=production" >> .env
echo "SECRET_KEY=$(openssl rand -hex 32)" >> .env

# Deploy
make deploy-docker

# Monitor
docker-compose logs -f cognitionsim
```

### Kubernetes Deployment

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/deployment.yaml

# Check status
kubectl get pods
kubectl get services

# Access the service
kubectl port-forward svc/cognitionsim-ai-service 5000:80
```

### Cloud Deployment Examples

#### AWS ECS

```bash
# Push image to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ECR_REGISTRY
docker tag cognitionsim-ai:latest YOUR_ECR_REGISTRY/cognitionsim-ai:latest
docker push YOUR_ECR_REGISTRY/cognitionsim-ai:latest

# Create ECS task definition and service
aws ecs create-service --cli-input-json file://ecs-service.json
```

#### Google Cloud Run

```bash
# Push to Google Container Registry
gcloud builds submit --tag gcr.io/YOUR_PROJECT/cognitionsim-ai

# Deploy
gcloud run deploy cognitionsim-ai \
  --image gcr.io/YOUR_PROJECT/cognitionsim-ai \
  --platform managed \
  --port 5000 \
  --memory 4Gi
```

#### Azure Container Instances

```bash
# Push to Azure Container Registry
az acr build --registry YOUR_REGISTRY --image cognitionsim-ai:latest .

# Deploy
az container create \
  --resource-group YOUR_RG \
  --name cognitionsim-ai \
  --image YOUR_REGISTRY.azurecr.io/cognitionsim-ai:latest \
  --cpu 2 --memory 4 \
  --ports 5000
```

## 🛠️ Makefile Commands

Common development and deployment tasks:

```bash
make help           # Show all available commands
make install        # Install dependencies
make dev-install    # Install dev dependencies
make test           # Run tests
make lint           # Run linting
make format         # Format code
make clean          # Clean build artifacts
make docker-build   # Build Docker image
make docker-run     # Run with Docker Compose
make train          # Run training locally
make dashboard      # Start web dashboard
make backup-state   # Backup model state
```

## 🔐 Security Best Practices

1. **Change SECRET_KEY in production**:
   ```bash
   export SECRET_KEY=$(openssl rand -hex 32)
   ```

2. **Use environment variables for secrets**:
   - Never commit `.env` file to Git
   - Use secrets management (AWS Secrets Manager, Azure Key Vault, etc.)

3. **Run as non-root user**:
   - Dockerfile already configured with user `quadra`

4. **Enable security scanning**:
   - Trivy scans included in CI/CD
   - Review security reports in GitHub Security tab

## 📊 Monitoring & Observability

### Health Checks

```bash
# Check application health
curl http://localhost:5000/health

# Response:
{
  "status": "healthy",
  "timestamp": "2025-12-21T01:00:00",
  "version": "1.0.0",
  "initialized": true,
  "running": false
}
```

### Logging

Configure log level via environment:

```bash
LOG_LEVEL=DEBUG python app.py
```

Access logs in Docker:

```bash
docker-compose logs -f cognitionsim
```

### Metrics Collection

The `/api/status` endpoint provides real-time metrics:

```bash
curl http://localhost:5000/api/status
```

## 🧪 Testing

### Run Tests Locally

```bash
# Install test dependencies
make dev-install

# Run tests
make test

# Run with coverage
pytest tests/ -v --cov=. --cov-report=html
```

### Test in Docker

```bash
# Build test image
docker build --target builder -t cognitionsim-test .

# Run tests
docker run cognitionsim-test pytest tests/
```

## 📈 Performance Optimization

### Resource Limits

Adjust in `docker-compose.yml` or Kubernetes manifests:

```yaml
resources:
  limits:
    memory: 4Gi
    cpu: 2000m
  requests:
    memory: 2Gi
    cpu: 1000m
```

### Scaling

**Docker Compose**:
```bash
docker-compose up -d --scale cognitionsim=3
```

**Kubernetes**:
```bash
kubectl scale deployment cognitionsim-ai --replicas=3
```

## 🔧 Troubleshooting

### Container Won't Start

```bash
# Check logs
docker-compose logs cognitionsim

# Check health
docker-compose ps
```

### Port Already in Use

```bash
# Change port in .env
HOST_PORT=5001

# Or use Docker directly
docker run -p 5001:5000 cognitionsim-ai
```

### Permission Denied

```bash
# Fix volume permissions
sudo chown -R $USER:$USER models/ dashboard_state/ logs/
```

### Out of Memory

```bash
# Reduce field size
echo "FIELD_SIZE=50" >> .env

# Increase Docker memory limit
# Docker Desktop: Settings → Resources → Memory
```

## 📚 Additional Resources

- [Flask Documentation](https://flask.palletsprojects.com/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)

## 🤝 Contributing

When adding new features:

1. Update tests in `tests/`
2. Run `make lint` and `make test`
3. Update documentation
4. Create pull request
5. CI/CD will validate changes

