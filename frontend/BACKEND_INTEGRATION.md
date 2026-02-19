# Mental Health Sentiment OS - Backend Integration Guide

This document provides complete instructions for connecting the frontend to your custom SLM (Small Language Model) via a FastAPI backend.

---

## 📋 Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [FastAPI Backend Setup](#fastapi-backend-setup)
3. [API Endpoint Specification](#api-endpoint-specification)
4. [Frontend Integration](#frontend-integration)
5. [Model Integration Examples](#model-integration-examples)
6. [Deployment Options](#deployment-options)
7. [Security Considerations](#security-considerations)

---

## 🏗️ Architecture Overview

```
┌─────────────────────┐     HTTP POST      ┌─────────────────────┐     Inference     ┌─────────────────────┐
│   React Frontend    │ ─────────────────► │   FastAPI Backend   │ ────────────────► │    Your SLM Model   │
│   (This Project)    │ ◄───────────────── │   (Python Server)   │ ◄──────────────── │   (HuggingFace/etc) │
└─────────────────────┘     JSON Response  └─────────────────────┘     Predictions   └─────────────────────┘
```

---

## 🚀 FastAPI Backend Setup

### Prerequisites

```bash
# Python 3.9+ required
python --version

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies

Create `requirements.txt`:

```txt
fastapi==0.109.0
uvicorn[standard]==0.27.0
pydantic==2.5.3
python-multipart==0.0.6
transformers==4.37.0
torch==2.1.2
sentencepiece==0.1.99
python-dotenv==1.0.0
```

Install:

```bash
pip install -r requirements.txt
```

### Project Structure

```
backend/
├── main.py              # FastAPI application
├── models/
│   ├── __init__.py
│   ├── sentiment.py     # Model loading & inference
│   └── schemas.py       # Pydantic models
├── routers/
│   ├── __init__.py
│   └── analyze.py       # Analysis endpoints
├── config.py            # Configuration
├── requirements.txt
└── .env                 # Environment variables
```

### Complete FastAPI Implementation

#### `main.py`

```python
"""
Mental Health Sentiment OS - FastAPI Backend
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import analyze
import uvicorn

app = FastAPI(
    title="Mental Health Sentiment API",
    description="API for sentiment analysis using SLM models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration - IMPORTANT for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",      # Vite dev server
        "http://localhost:8080",      # Alternative port
        "https://your-deployed-frontend.com"  # Production URL
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(analyze.router, prefix="/api/v1", tags=["Analysis"])

@app.get("/")
async def root():
    return {"message": "Mental Health Sentiment API", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
```

#### `models/schemas.py`

```python
"""
Pydantic schemas for request/response validation
"""
from pydantic import BaseModel, Field
from typing import Optional, List

class AnalysisRequest(BaseModel):
    """Request schema for sentiment analysis"""
    text: str = Field(
        ..., 
        min_length=1, 
        max_length=5000,
        description="Text to analyze for mental health indicators",
        example="I've been feeling really overwhelmed with work lately, but I'm trying to stay positive."
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Today was challenging but I managed to complete my tasks and felt accomplished."
            }
        }

class SentimentMetric(BaseModel):
    """Individual sentiment metric with dynamic label"""
    label: str = Field(..., description="Display label for the metric (e.g., 'Anxiety Indicators')")
    value: int = Field(..., ge=0, le=100, description="Score value (0-100)")

class SentimentResults(BaseModel):
    """Container for all sentiment metrics"""
    metrics: List[SentimentMetric] = Field(..., description="List of sentiment metrics")

class AnalysisResponse(BaseModel):
    """Response schema for sentiment analysis"""
    success: bool
    data: Optional[SentimentResults] = None
    message: Optional[str] = None
    model_version: str = "1.0.0"
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "data": {
                    "metrics": [
                        {"label": "Anxiety Indicators", "value": 35},
                        {"label": "Mood Stability", "value": 72},
                        {"label": "Stress Level", "value": 45}
                    ]
                },
                "message": "Analysis completed successfully",
                "model_version": "1.0.0"
            }
        }
```

#### `models/sentiment.py`

```python
"""
Sentiment analysis model loader and inference
"""
import os
from typing import Dict
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch

class SentimentAnalyzer:
    """
    Wrapper class for your SLM model.
    Replace the model loading with your specific model.
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialize the sentiment analyzer.
        
        Args:
            model_name: HuggingFace model name or local path
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name or os.getenv("MODEL_NAME", "distilbert-base-uncased-finetuned-sst-2-english")
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load the model and tokenizer"""
        print(f"Loading model: {self.model_name} on {self.device}")
        
        # Option 1: Use HuggingFace pipeline (simpler)
        self.pipeline = pipeline(
            "text-classification",
            model=self.model_name,
            device=0 if self.device == "cuda" else -1,
            top_k=None  # Return all scores
        )
        
        # Option 2: Load model directly (more control)
        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        # self.model.to(self.device)
        
        print("Model loaded successfully!")
    
    def analyze(self, text: str) -> Dict:
        """
        Analyze text and return sentiment metrics.
        
        This is where you implement your custom logic to convert
        model outputs to dynamic metrics with labels.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dict with metrics array containing label/value pairs
        """
        # Get raw model output
        result = self.pipeline(text)
        
        # Example: Convert model output to metrics
        # This logic depends on your specific model's output format
        
        positive_score = 0
        negative_score = 0
        
        for item in result[0]:
            if item['label'].lower() in ['positive', 'pos', 'label_1']:
                positive_score = item['score']
            elif item['label'].lower() in ['negative', 'neg', 'label_0']:
                negative_score = item['score']
        
        # Derive metrics (replace with your model's logic)
        # Labels can be dynamic based on your model or text content
        anxiety = int(negative_score * 100 * 0.8)
        mood = int(positive_score * 100)
        stress = int((negative_score * 0.6 + (1 - positive_score) * 0.4) * 100)
        
        return {
            "metrics": [
                {"label": "Anxiety Indicators", "value": min(max(anxiety, 0), 100)},
                {"label": "Mood Stability", "value": min(max(mood, 0), 100)},
                {"label": "Stress Level", "value": min(max(stress, 0), 100)}
            ]
        }

# Singleton instance
_analyzer = None

def get_analyzer() -> SentimentAnalyzer:
    """Get or create the sentiment analyzer instance"""
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentAnalyzer()
    return _analyzer
```

#### `routers/analyze.py`

```python
"""
Analysis endpoints
"""
from fastapi import APIRouter, HTTPException, Depends
from models.schemas import AnalysisRequest, AnalysisResponse, SentimentScores
from models.sentiment import get_analyzer, SentimentAnalyzer

router = APIRouter()

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_text(
    request: AnalysisRequest,
    analyzer: SentimentAnalyzer = Depends(get_analyzer)
):
    """
    Analyze text for mental health sentiment indicators.
    
    - **text**: The text to analyze (1-5000 characters)
    
    Returns anxiety, mood stability, and stress level scores (0-100).
    """
    try:
        # Perform analysis
        scores = analyzer.analyze(request.text)
        
        return AnalysisResponse(
            success=True,
            data=SentimentScores(**scores),
            message="Analysis completed successfully"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

@router.get("/models/info")
async def get_model_info(
    analyzer: SentimentAnalyzer = Depends(get_analyzer)
):
    """Get information about the loaded model"""
    return {
        "model_name": analyzer.model_name,
        "device": analyzer.device,
        "status": "loaded"
    }
```

#### `config.py`

```python
"""
Configuration management
"""
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings"""
    
    # API Configuration
    api_title: str = "Mental Health Sentiment API"
    api_version: str = "1.0.0"
    debug: bool = False
    
    # Model Configuration
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"
    model_cache_dir: str = "./model_cache"
    
    # CORS
    allowed_origins: list = [
        "http://localhost:5173",
        "http://localhost:8080"
    ]
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings() -> Settings:
    return Settings()
```

#### `.env` (Example)

```env
# Model Configuration
MODEL_NAME=distilbert-base-uncased-finetuned-sst-2-english
# For custom models:
# MODEL_NAME=./path/to/your/local/model
# MODEL_NAME=your-huggingface-username/your-model-name

# API Configuration
DEBUG=true
```

### Running the Backend

```bash
# Development mode with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

Visit `http://localhost:8000/docs` for interactive API documentation.

---

## 🔌 Frontend Integration

### Update the SentimentWindow Component

Replace the mock `simulateAnalysis` function with real API calls:

```typescript
// src/hooks/useSentimentAnalysis.ts

import { useState } from 'react';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface SentimentMetric {
  label: string;
  value: number;
}

export interface SentimentResults {
  metrics: SentimentMetric[];
}

interface AnalysisResponse {
  success: boolean;
  data: SentimentResults | null;
  message: string;
  model_version: string;
}

export function useSentimentAnalysis() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<SentimentResults | null>(null);

  const analyze = async (text: string): Promise<SentimentResults | null> => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: AnalysisResponse = await response.json();

      if (!data.success || !data.data) {
        throw new Error(data.message || 'Analysis failed');
      }

      setResults(data.data);
      return data.data;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'An error occurred';
      setError(message);
      return null;
    } finally {
      setIsLoading(false);
    }
  };

  const reset = () => {
    setResults(null);
    setError(null);
  };

  return { analyze, isLoading, error, results, setResults, reset };
}
```

### Environment Variables

Create `.env` in your frontend project:

```env
VITE_API_URL=http://localhost:8000
```

### Updated SentimentWindow Component

```typescript
// Update src/components/SentimentWindow.tsx

import { useSentimentAnalysis } from '@/hooks/useSentimentAnalysis';
import { toast } from 'sonner';

// In the component:
const { analyze, isLoading, error, results, setResults } = useSentimentAnalysis();

const handleSubmit = async () => {
  if (!inputText.trim()) return;
  
  setResults(null);
  
  const analysisResults = await analyze(inputText);
  
  if (analysisResults) {
    // Trigger confetti if mood metric is high
    const moodMetric = analysisResults.metrics.find(m => 
      m.label.toLowerCase().includes('mood')
    );
    if (moodMetric && moodMetric.value >= 70) {
      triggerSparkles();
    }
  } else if (error) {
    toast.error('Analysis failed. Please try again.');
  }
};
```

---

## 🤖 Model Integration Examples

### Example 1: Using a HuggingFace Model

```python
# For emotion detection
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

# Custom analysis logic for this model
def analyze(self, text: str) -> Dict:
    result = self.pipeline(text)
    
    # This model returns: anger, disgust, fear, joy, neutral, sadness, surprise
    emotions = {r['label']: r['score'] for r in result[0]}
    
    anxiety = int((emotions.get('fear', 0) + emotions.get('surprise', 0) * 0.3) * 100)
    mood = int((emotions.get('joy', 0) + emotions.get('neutral', 0) * 0.5) * 100)
    stress = int((emotions.get('anger', 0) + emotions.get('sadness', 0) + emotions.get('disgust', 0) * 0.5) * 100)
    
    # Return with dynamic labels - you can customize these based on analysis
    return {
        "metrics": [
            {"label": "Anxiety Indicators", "value": anxiety},
            {"label": "Mood Stability", "value": mood},
            {"label": "Stress Level", "value": stress}
        ]
    }
```

### Example 2: Using Your Own Fine-tuned Model

```python
# Load from local path
MODEL_NAME = "./my_finetuned_model"

# Or from HuggingFace Hub after uploading
MODEL_NAME = "your-username/mental-health-sentiment-model"
```

### Example 3: Using OpenAI-compatible API

```python
import openai

class SentimentAnalyzer:
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def analyze(self, text: str) -> Dict:
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": """Analyze the following text for mental health indicators.
                Return a JSON object with a "metrics" array. Each metric should have:
                - label: descriptive name for the metric
                - value: score from 0-100
                
                Include relevant metrics based on the text content. Example metrics:
                Anxiety Indicators, Mood Stability, Stress Level, Hope Level, Energy, etc.
                
                Only return the JSON, no other text.
                Example: {"metrics": [{"label": "Anxiety Indicators", "value": 35}]}"""},
                {"role": "user", "content": text}
            ],
            temperature=0.3
        )
        
        import json
        return json.loads(response.choices[0].message.content)
```

---

## 🚢 Deployment Options

### Option 1: Docker

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_NAME=distilbert-base-uncased-finetuned-sst-2-english
    volumes:
      - ./model_cache:/app/model_cache
```

### Option 2: Railway / Render / Fly.io

These platforms auto-detect Python projects. Just push your code and set environment variables.

### Option 3: AWS Lambda with Mangum

```python
from mangum import Mangum
handler = Mangum(app)
```

---

## 🔒 Security Considerations

1. **Rate Limiting**: Add rate limiting to prevent abuse

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@router.post("/analyze")
@limiter.limit("10/minute")
async def analyze_text(request: Request, ...):
    ...
```

2. **Input Validation**: Already handled by Pydantic schemas

3. **CORS**: Restrict origins in production

4. **HTTPS**: Use a reverse proxy (nginx, Caddy) with SSL in production

5. **API Keys**: For production, add authentication

```python
from fastapi.security import HTTPBearer
security = HTTPBearer()

@router.post("/analyze")
async def analyze_text(
    request: AnalysisRequest,
    token: str = Depends(security)
):
    # Validate token
    ...
```

---

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [Pydantic V2 Documentation](https://docs.pydantic.dev/)
- [Uvicorn Documentation](https://www.uvicorn.org/)

---

## 🆘 Troubleshooting

**CORS Errors**: Ensure your frontend URL is in `allow_origins`

**Model Loading Slow**: First load downloads the model. Set `model_cache_dir` to persist.

**GPU Not Detected**: Install CUDA-compatible PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

**Memory Issues**: Use smaller models or quantized versions for limited hardware.

---

*Built with ❤️ for portfolio demonstration purposes. Not for medical use.*
