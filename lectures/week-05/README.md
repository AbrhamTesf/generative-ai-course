# Week 5: API Development & Deployment

## 📚 Lecture Overview

This week we focus on building production-ready APIs for AI models. You'll learn how to design, implement, and deploy scalable API services using FastAPI and Flask, with proper error handling, authentication, and monitoring.

## 🎯 Learning Objectives

By the end of this week, you will:
- Design and implement RESTful APIs for AI models
- Master FastAPI and Flask frameworks
- Implement proper authentication and security
- Deploy APIs using Docker and cloud platforms
- Monitor and optimize API performance

## 📖 Lecture Content

### 1. API Design Principles

**RESTful API Design:**
- **Resources**: Nouns, not verbs
- **HTTP Methods**: GET, POST, PUT, DELETE
- **Status Codes**: Proper HTTP response codes
- **Versioning**: API versioning strategies

**API Design Patterns:**
```python
# Good API Design
GET /api/v1/models          # List models
GET /api/v1/models/{id}     # Get specific model
POST /api/v1/models         # Create new model
PUT /api/v1/models/{id}     # Update model
DELETE /api/v1/models/{id}  # Delete model

# Bad API Design
GET /api/v1/getModels
POST /api/v1/createModel
POST /api/v1/updateModel
POST /api/v1/deleteModel
```

### 2. FastAPI Framework

**FastAPI Setup:**
```python
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn

app = FastAPI(
    title="AI Model API",
    description="API for AI model inference",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class TextRequest(BaseModel):
    text: str
    max_length: Optional[int] = 100
    temperature: Optional[float] = 0.7

class TextResponse(BaseModel):
    generated_text: str
    model_used: str
    processing_time: float

class ErrorResponse(BaseModel):
    error: str
    detail: str
```

**Basic API Endpoints:**
```python
@app.get("/")
async def root():
    return {"message": "AI Model API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now()}

@app.get("/models")
async def list_models():
    """List available models"""
    return {
        "models": [
            {"id": "gpt-3.5", "name": "GPT-3.5", "type": "text-generation"},
            {"id": "stable-diffusion", "name": "Stable Diffusion", "type": "image-generation"}
        ]
    }

@app.post("/generate/text", response_model=TextResponse)
async def generate_text(request: TextRequest):
    """Generate text using AI model"""
    try:
        start_time = time.time()
        
        # Model inference
        generated_text = await generate_text_with_model(
            request.text, 
            request.max_length, 
            request.temperature
        )
        
        processing_time = time.time() - start_time
        
        return TextResponse(
            generated_text=generated_text,
            model_used="gpt-3.5",
            processing_time=processing_time
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### 3. Advanced FastAPI Features

**Dependencies and Authentication:**
```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

security = HTTPBearer()

# JWT token validation
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(
            credentials.credentials, 
            SECRET_KEY, 
            algorithms=["HS256"]
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

# Protected endpoint
@app.post("/protected/generate")
async def protected_generate(
    request: TextRequest,
    user: dict = Depends(verify_token)
):
    """Protected endpoint requiring authentication"""
    return await generate_text(request)
```

**File Upload and Processing:**
```python
from fastapi import File, UploadFile
from PIL import Image
import io

@app.post("/upload/image")
async def upload_image(file: UploadFile = File(...)):
    """Upload and process image"""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Read and process image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    # Process with AI model
    result = await process_image_with_ai(image)
    
    return {"result": result, "filename": file.filename}
```

**Background Tasks:**
```python
from fastapi import BackgroundTasks
import asyncio

@app.post("/generate/async")
async def generate_async(
    request: TextRequest,
    background_tasks: BackgroundTasks
):
    """Generate text asynchronously"""
    task_id = generate_task_id()
    
    # Add task to background queue
    background_tasks.add_task(
        process_generation_task,
        task_id,
        request
    )
    
    return {"task_id": task_id, "status": "processing"}

@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Get task status"""
    status = await get_task_status_from_db(task_id)
    return {"task_id": task_id, "status": status}
```

### 4. Flask Framework

**Flask API Setup:**
```python
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

app = Flask(__name__)
CORS(app)

# Rate limiting
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

# Routes
@app.route("/api/v1/generate", methods=["POST"])
@limiter.limit("10 per minute")
def generate_text():
    """Generate text endpoint"""
    try:
        data = request.get_json()
        text = data.get("text", "")
        max_length = data.get("max_length", 100)
        
        if not text:
            return jsonify({"error": "Text is required"}), 400
        
        # Model inference
        result = generate_with_model(text, max_length)
        
        return jsonify({
            "generated_text": result,
            "status": "success"
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
```

### 5. Model Serving and Optimization

**Model Loading and Caching:**
```python
import torch
from transformers import AutoModel, AutoTokenizer
import redis
import pickle

class ModelManager:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
    
    def load_model(self, model_name: str):
        """Load model with caching"""
        if model_name in self.models:
            return self.models[model_name]
        
        # Load from cache if available
        cached_model = self.redis_client.get(f"model:{model_name}")
        if cached_model:
            model = pickle.loads(cached_model)
            self.models[model_name] = model
            return model
        
        # Load from Hugging Face
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Cache model
        self.redis_client.setex(
            f"model:{model_name}",
            3600,  # 1 hour cache
            pickle.dumps(model)
        )
        
        self.models[model_name] = model
        self.tokenizers[model_name] = tokenizer
        
        return model
    
    def get_tokenizer(self, model_name: str):
        """Get tokenizer for model"""
        if model_name not in self.tokenizers:
            self.load_model(model_name)
        return self.tokenizers[model_name]

# Global model manager
model_manager = ModelManager()
```

**Batch Processing:**
```python
from concurrent.futures import ThreadPoolExecutor
import asyncio

class BatchProcessor:
    def __init__(self, max_workers=4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.batch_size = 32
        self.batch_timeout = 1.0  # seconds
    
    async def process_batch(self, requests):
        """Process multiple requests in batch"""
        loop = asyncio.get_event_loop()
        
        # Group requests into batches
        batches = [requests[i:i + self.batch_size] 
                  for i in range(0, len(requests), self.batch_size)]
        
        results = []
        for batch in batches:
            # Process batch in thread pool
            batch_result = await loop.run_in_executor(
                self.executor,
                self._process_single_batch,
                batch
            )
            results.extend(batch_result)
        
        return results
    
    def _process_single_batch(self, batch):
        """Process a single batch of requests"""
        # Implement batch processing logic
        return [self._process_single_request(req) for req in batch]

batch_processor = BatchProcessor()
```

### 6. Docker and Containerization

**Dockerfile for AI API:**
```dockerfile
# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Docker Compose:**
```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379
      - MODEL_CACHE_DIR=/app/models
    volumes:
      - model_cache:/app/models
    depends_on:
      - redis
    restart: unless-stopped

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - api
    restart: unless-stopped

volumes:
  model_cache:
  redis_data:
```

### 7. Monitoring and Logging

**Structured Logging:**
```python
import logging
import json
from datetime import datetime
from fastapi import Request
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StructuredLogger:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def log_request(self, request: Request, response_time: float):
        """Log request details"""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "method": request.method,
            "url": str(request.url),
            "client_ip": request.client.host,
            "response_time": response_time,
            "user_agent": request.headers.get("user-agent", "")
        }
        
        self.logger.info(json.dumps(log_data))
    
    def log_error(self, error: Exception, context: dict = None):
        """Log error details"""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(error),
            "error_type": type(error).__name__,
            "context": context or {}
        }
        
        self.logger.error(json.dumps(log_data))

# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    response_time = time.time() - start_time
    
    logger.log_request(request, response_time)
    return response
```

**Performance Monitoring:**
```python
from prometheus_client import Counter, Histogram, generate_latest
import psutil

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
MODEL_INFERENCE_TIME = Histogram('model_inference_seconds', 'Model inference time')

@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path).inc()
    REQUEST_DURATION.observe(duration)
    
    return response

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type="text/plain")
```

## 🛠️ Workshop Preparation

**Pre-workshop Tasks:**
1. Install FastAPI and Flask: `pip install fastapi uvicorn flask flask-cors`
2. Set up Docker environment
3. Prepare sample AI models for serving
4. Review API design best practices

**Workshop Goals:**
- Build a complete AI model API
- Implement authentication and rate limiting
- Containerize the application
- Deploy to cloud platform

## 📚 Additional Resources

**Reading Materials:**
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [REST API Design](https://restfulapi.net/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)

**Videos:**
- [FastAPI Tutorial](https://www.youtube.com/watch?v=0hUkJscqeN0)
- [API Design](https://www.youtube.com/watch?v=lsMQRaeKNDk)
- [Docker for AI](https://www.youtube.com/watch?v=YQpV7WVTJXE)

**Tools & Platforms:**
- [Postman](https://www.postman.com/) - API testing
- [Swagger UI](https://swagger.io/tools/swagger-ui/) - API documentation
- [Prometheus](https://prometheus.io/) - Monitoring
- [Grafana](https://grafana.com/) - Visualization

## 📝 Assignment

**Due Date**: End of Week 5

**Tasks:**
1. Build a complete AI model API
2. Implement authentication and security
3. Add monitoring and logging
4. Deploy using Docker

**Submission Requirements:**
- Working API with documentation
- Docker configuration
- Security implementation
- Performance analysis

## 🎯 Next Week Preview

**Week 6: Frontend Integration & User Experience**
- Frontend frameworks and technologies
- Real-time communication
- User interface design
- Performance optimization

## 💡 Advanced Concepts

**API Gateway Patterns:**
- Rate limiting and throttling
- Circuit breakers
- Service discovery
- Load balancing

**Microservices Architecture:**
- Service decomposition
- Inter-service communication
- Data consistency
- Deployment strategies

## 🚀 Best Practices

**Security:**
- Input validation and sanitization
- Rate limiting and DDoS protection
- API key management
- HTTPS and encryption

**Performance:**
- Caching strategies
- Database optimization
- CDN integration
- Load testing

**Monitoring:**
- Health checks and alerts
- Performance metrics
- Error tracking
- User analytics

## 📞 Support

- **Element Support**: `#support`
- **Office Hours**: TBD
- **Email**: course-instructor@example.com

---

*Remember: A well-designed API is the foundation of any successful AI application. Focus on reliability, security, and performance!* 