# Week 7: Automation & CI/CD Pipelines

## 📚 Lecture Overview

This week we dive into DevOps and automation for AI applications. You'll learn how to set up continuous integration and deployment pipelines using GitHub Actions, implement automated testing, and deploy your applications with confidence.

## 🎯 Learning Objectives

By the end of this week, you will:
- Set up GitHub Actions workflows for AI applications
- Implement automated testing and quality checks
- Deploy applications using CI/CD pipelines
- Monitor and maintain production systems
- Apply DevOps best practices

## 📖 Lecture Content

### 1. Introduction to CI/CD

**What is CI/CD?**
- **Continuous Integration**: Automatically build and test code changes
- **Continuous Deployment**: Automatically deploy to production
- **Benefits**: Faster delivery, reduced errors, improved collaboration

**CI/CD Pipeline Stages:**
```
Code Commit → Build → Test → Security Scan → Deploy → Monitor
```

### 2. GitHub Actions Workflows

**Basic Workflow Structure:**
```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Run tests
        run: |
          pytest --cov=./ --cov-report=xml
```

**AI-Specific Workflow:**
```yaml
name: AI Model CI/CD

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

env:
  PYTHON_VERSION: '3.11'
  CUDA_VERSION: '11.8'

jobs:
  test:
    runs-on: ubuntu-latest
    
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Cache pip dependencies
        uses: actions/cache@v3
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
          restore-keys: |
            ${{ runner.os }}-pip-
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest pytest-cov flake8 black mypy
      
      - name: Lint with flake8
        run: |
          flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
          flake8 . --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics
      
      - name: Check formatting with black
        run: |
          black --check --diff .
      
      - name: Type check with mypy
        run: |
          mypy . --ignore-missing-imports
      
      - name: Run tests
        run: |
          pytest --cov=./ --cov-report=xml --cov-report=term-missing
      
      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
          flags: unittests
          name: codecov-umbrella
          fail_ci_if_error: true

  security:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install security tools
        run: |
          pip install bandit safety
      
      - name: Run security checks
        run: |
          bandit -r . -f json -o bandit-report.json || true
          safety check --json --output safety-report.json || true
      
      - name: Upload security reports
        uses: actions/upload-artifact@v3
        with:
          name: security-reports
          path: |
            bandit-report.json
            safety-report.json

  build:
    runs-on: ubuntu-latest
    needs: [test, security]
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
      
      - name: Build Docker image
        run: |
          docker build -t ai-app:${{ github.sha }} .
      
      - name: Upload Docker image
        uses: actions/upload-artifact@v3
        with:
          name: docker-image
          path: /tmp/docker-image.tar

  deploy:
    runs-on: ubuntu-latest
    needs: [build]
    if: github.ref == 'refs/heads/main'
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Deploy to staging
        run: |
          echo "Deploying to staging environment..."
          # Add your deployment commands here
      
      - name: Run smoke tests
        run: |
          echo "Running smoke tests..."
          # Add your smoke test commands here
```

### 3. Automated Testing

**Unit Tests:**
```python
import pytest
from unittest.mock import Mock, patch
from your_app.models import AIModel
from your_app.api import generate_text

class TestAIModel:
    def test_model_initialization(self):
        """Test model initialization"""
        model = AIModel("gpt-3.5-turbo")
        assert model.model_name == "gpt-3.5-turbo"
        assert model.temperature == 0.7
    
    def test_text_generation(self):
        """Test text generation"""
        model = AIModel("gpt-3.5-turbo")
        with patch('openai.ChatCompletion.create') as mock_create:
            mock_create.return_value = Mock(
                choices=[Mock(message=Mock(content="Generated text"))]
            )
            
            result = model.generate("Hello")
            assert result == "Generated text"
    
    def test_invalid_input(self):
        """Test handling of invalid input"""
        model = AIModel("gpt-3.5-turbo")
        with pytest.raises(ValueError):
            model.generate("")

class TestAPI:
    def test_generate_text_endpoint(self, client):
        """Test text generation endpoint"""
        response = client.post("/api/generate", json={
            "text": "Hello",
            "max_length": 50
        })
        
        assert response.status_code == 200
        assert "generated_text" in response.json()
    
    def test_invalid_request(self, client):
        """Test invalid request handling"""
        response = client.post("/api/generate", json={
            "text": ""
        })
        
        assert response.status_code == 400
        assert "error" in response.json()
```

**Integration Tests:**
```python
import pytest
from fastapi.testclient import TestClient
from your_app.main import app

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def auth_headers():
    return {"Authorization": "Bearer test-token"}

class TestIntegration:
    def test_full_text_generation_flow(self, client, auth_headers):
        """Test complete text generation flow"""
        # Test authentication
        response = client.get("/api/health", headers=auth_headers)
        assert response.status_code == 200
        
        # Test text generation
        response = client.post("/api/generate", 
                             json={"text": "Hello", "max_length": 50},
                             headers=auth_headers)
        assert response.status_code == 200
        
        # Test response format
        data = response.json()
        assert "generated_text" in data
        assert "processing_time" in data
        assert isinstance(data["generated_text"], str)
    
    def test_file_upload_flow(self, client, auth_headers):
        """Test file upload and processing flow"""
        # Create test file
        test_file = ("test.txt", b"Test content", "text/plain")
        
        response = client.post("/api/upload",
                             files={"file": test_file},
                             headers=auth_headers)
        
        assert response.status_code == 200
        assert "file_id" in response.json()
```

**Performance Tests:**
```python
import pytest
import time
from locust import HttpUser, task, between

class AIAppUser(HttpUser):
    wait_time = between(1, 3)
    
    @task(3)
    def generate_text(self):
        """Test text generation performance"""
        start_time = time.time()
        
        response = self.client.post("/api/generate", json={
            "text": "Hello, how are you?",
            "max_length": 100
        })
        
        response_time = time.time() - start_time
        
        assert response.status_code == 200
        assert response_time < 5.0  # Should complete within 5 seconds
    
    @task(1)
    def upload_file(self):
        """Test file upload performance"""
        files = {"file": ("test.txt", b"Test content", "text/plain")}
        
        response = self.client.post("/api/upload", files=files)
        
        assert response.status_code == 200
```

### 4. Security Scanning

**Automated Security Checks:**
```yaml
- name: Run security scans
  run: |
    # Bandit for Python security issues
    bandit -r . -f json -o bandit-report.json || true
    
    # Safety for dependency vulnerabilities
    safety check --json --output safety-report.json || true
    
    # Semgrep for code security patterns
    semgrep ci --json --output semgrep-report.json || true

- name: Upload security reports
  uses: actions/upload-artifact@v3
  with:
    name: security-reports
    path: |
      bandit-report.json
      safety-report.json
      semgrep-report.json
```

**Dependency Scanning:**
```python
# requirements-check.py
import subprocess
import json
import sys

def check_dependencies():
    """Check for vulnerable dependencies"""
    try:
        result = subprocess.run(
            ["safety", "check", "--json"],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            vulnerabilities = json.loads(result.stdout)
            print("Vulnerabilities found:")
            for vuln in vulnerabilities:
                print(f"- {vuln['package']}: {vuln['vulnerability_id']}")
            sys.exit(1)
        else:
            print("No vulnerabilities found")
            
    except Exception as e:
        print(f"Error checking dependencies: {e}")
        sys.exit(1)

if __name__ == "__main__":
    check_dependencies()
```

### 5. Docker and Containerization

**Multi-stage Dockerfile:**
```dockerfile
# Build stage
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Production stage
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Copy application code
COPY --chown=appuser:appuser . .

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Docker Compose for Development:**
```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/ai_app
      - REDIS_URL=redis://redis:6379
    volumes:
      - .:/app
      - model_cache:/app/models
    depends_on:
      - db
      - redis
    restart: unless-stopped

  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=ai_app
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - app

volumes:
  postgres_data:
  redis_data:
  model_cache:
```

### 6. Deployment Strategies

**Blue-Green Deployment:**
```yaml
- name: Blue-Green Deployment
  run: |
    # Deploy new version (green)
    kubectl apply -f k8s/green-deployment.yaml
    
    # Wait for green deployment to be ready
    kubectl rollout status deployment/ai-app-green
    
    # Switch traffic to green
    kubectl apply -f k8s/green-service.yaml
    
    # Run smoke tests
    ./scripts/smoke-tests.sh
    
    # If tests pass, remove blue deployment
    kubectl delete -f k8s/blue-deployment.yaml
```

**Canary Deployment:**
```yaml
- name: Canary Deployment
  run: |
    # Deploy canary with 10% traffic
    kubectl apply -f k8s/canary-deployment.yaml
    
    # Monitor canary performance
    ./scripts/monitor-canary.sh
    
    # If successful, gradually increase traffic
    kubectl patch service ai-app-service -p '{"spec":{"selector":{"version":"canary"}}}'
```

### 7. Monitoring and Observability

**Application Monitoring:**
```python
import logging
import time
from functools import wraps
from prometheus_client import Counter, Histogram, generate_latest
from fastapi import Request, Response

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
MODEL_INFERENCE_TIME = Histogram('model_inference_seconds', 'Model inference time')

# Structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def monitor_request(request: Request, response: Response, duration: float):
    """Monitor request metrics"""
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path
    ).inc()
    
    REQUEST_DURATION.observe(duration)
    
    logger.info(f"Request {request.method} {request.url.path} - {response.status_code} - {duration:.3f}s")

# Middleware
@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    monitor_request(request, response, duration)
    return response

# Metrics endpoint
@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

**Health Checks:**
```python
@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {}
    }
    
    # Check database
    try:
        db.execute("SELECT 1")
        health_status["services"]["database"] = "healthy"
    except Exception as e:
        health_status["services"]["database"] = f"unhealthy: {e}"
        health_status["status"] = "unhealthy"
    
    # Check Redis
    try:
        redis.ping()
        health_status["services"]["redis"] = "healthy"
    except Exception as e:
        health_status["services"]["redis"] = f"unhealthy: {e}"
        health_status["status"] = "unhealthy"
    
    # Check AI model
    try:
        model.predict("test")
        health_status["services"]["ai_model"] = "healthy"
    except Exception as e:
        health_status["services"]["ai_model"] = f"unhealthy: {e}"
        health_status["status"] = "unhealthy"
    
    status_code = 200 if health_status["status"] == "healthy" else 503
    return JSONResponse(content=health_status, status_code=status_code)
```

## 🛠️ Workshop Preparation

**Pre-workshop Tasks:**
1. Set up GitHub repository with Actions
2. Install Docker and Docker Compose
3. Review testing frameworks (pytest, unittest)
4. Set up monitoring tools (Prometheus, Grafana)

**Workshop Goals:**
- Set up complete CI/CD pipeline
- Implement automated testing
- Deploy application with Docker
- Monitor application performance

## 📚 Additional Resources

**Reading Materials:**
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Testing Python Applications](https://docs.pytest.org/)
- [DevOps Handbook](https://itrevolution.com/the-devops-handbook/)

**Videos:**
- [GitHub Actions Tutorial](https://www.youtube.com/watch?v=R8_veQiYBjI)
- [Docker for Beginners](https://www.youtube.com/watch?v=3c-iBn73dDE)
- [CI/CD Pipeline](https://www.youtube.com/watch?v=1er2cjUq1UI)

**Tools & Platforms:**
- [GitHub Actions](https://github.com/features/actions)
- [Docker Hub](https://hub.docker.com/)
- [Prometheus](https://prometheus.io/)
- [Grafana](https://grafana.com/)

## 📝 Assignment

**Due Date**: End of Week 7

**Tasks:**
1. Set up complete CI/CD pipeline
2. Implement comprehensive testing
3. Deploy application with monitoring
4. Document deployment process

**Submission Requirements:**
- Working CI/CD pipeline
- Test coverage report
- Deployment documentation
- Monitoring setup

## 🎯 Next Week Preview

**Week 8: Capstone Project Development**
- Project planning and architecture
- Integration of all course concepts
- Final project development
- Presentation and demonstration

## 💡 Advanced Concepts

**Infrastructure as Code:**
- Terraform and CloudFormation
- Kubernetes manifests
- Configuration management
- Environment management

**Advanced Monitoring:**
- Distributed tracing
- Log aggregation
- Alert management
- Performance optimization

## 🚀 Best Practices

**CI/CD:**
- Keep pipelines fast and reliable
- Use feature flags for deployments
- Implement rollback strategies
- Monitor pipeline performance

**Testing:**
- Write tests for all critical paths
- Use test data management
- Implement performance testing
- Maintain test documentation

**Security:**
- Scan dependencies regularly
- Implement security gates
- Use secrets management
- Monitor for vulnerabilities

## 📞 Support

- **Element Support**: `#support`
- **Office Hours**: TBD
- **Email**: course-instructor@example.com

---

*Remember: Automation is the key to reliable, scalable AI applications. Build it once, deploy it everywhere!* 