---
title: Datich Mental State Analyzer
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
---

# Datich Mental State Analyzer API

A FastAPI-based mental state analysis API using a fine-tuned Qwen2.5-0.5B model with LoRA adapters.

## API Endpoints

### Health Check
- `GET /` - Root endpoint
- `GET /health` - Health check

### Analysis
- `POST /api/v1/analyze` - Analyze text for mental state indicators

**Request Body:**
```json
{
  "text": "Your text to analyze here..."
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "scores": {
      "sadness": 0.65,
      "anxiety": 0.42,
      "rumination": 0.38,
      "self_focus": 0.55,
      "hopelessness": 0.71,
      "emotional_volatility": 0.33
    },
    "classification": {
      "primary_profile": "Severe Distress / Depressive Profile",
      "top_3_matches": [
        {"profile": "Severe Distress / Depressive Profile", "confidence_percentage": 45.2},
        {"profile": "Passive Sadness / Apathy", "confidence_percentage": 28.1},
        {"profile": "Baseline / Mild Anxiety", "confidence_percentage": 15.3}
      ]
    }
  },
  "message": "Analysis completed successfully",
  "model_version": "1.0.0"
}
```

### Model Info
- `GET /api/v1/models/info` - Get model information

## Interactive Docs

Visit `/docs` for Swagger UI or `/redoc` for ReDoc documentation.
