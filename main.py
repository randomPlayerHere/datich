from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import torch
import numpy as np
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import os

# ============== Pydantic Schemas ==============

class AnalysisRequest(BaseModel):
    text: str = Field(
        ..., 
        min_length=1, 
        max_length=5000,
        description="The text to analyze",
    )

class EmotionScores(BaseModel):
    sadness: float
    anxiety: float
    rumination: float
    self_focus: float
    hopelessness: float
    emotional_volatility: float

class ProfileMatch(BaseModel):
    profile: str
    confidence_percentage: float

class Classification(BaseModel):
    primary_profile: str
    top_3_matches: List[ProfileMatch]

class AnalysisData(BaseModel):
    scores: EmotionScores
    classification: Classification

class AnalysisResponse(BaseModel):
    success: bool
    data: Optional[AnalysisData] = None
    message: Optional[str] = None
    model_version: str = "1.0.0"

# ============== Model Loading ==============

print("Loading model...")

# Paths - adjust based on your HF Space structure
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ADAPTER_PATH = os.path.join(BASE_DIR, "ml/binaries/qwen_lora")
SCALER_PATH = os.path.join(BASE_DIR, "ml/binaries/datich_scaler.pkl")
KMEANS_PATH = os.path.join(BASE_DIR, "ml/binaries/datich_kmeans_model.pkl")

BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
base_model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL_ID,
    num_labels=6,
    problem_type="regression",
    device_map="auto",
    torch_dtype=torch.float32  # Use float32 for CPU compatibility
)
base_model.config.pad_token_id = tokenizer.pad_token_id
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

# Load sklearn models
scaler = joblib.load(SCALER_PATH)
kmeans = joblib.load(KMEANS_PATH)

print("Model loaded successfully!")

# ============== Constants ==============

EMOTIONS = ['sadness', 'anxiety', 'rumination', 'self_focus', 'hopelessness', 'emotional_volatility']

CLUSTER_MAPPING = {
    0: "Severe Distress / Depressive Profile",
    1: "Passive Sadness / Apathy",
    2: "Baseline / Mild Anxiety",
    3: "Emotionally Volatile / Dysregulated"
}

# ============== Prediction Function ==============

def predict_mental_state(text: str) -> Dict:
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512, padding=True
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        raw_scores = outputs.logits[0].float().cpu().numpy()
    
    clipped_scores = np.clip(raw_scores, 0.0, 1.0)
    
    scores_dict = {
        emotion: round(float(score), 3) 
        for emotion, score in zip(EMOTIONS, clipped_scores)
    }
    
    features_array = np.array([[scores_dict[feat] for feat in EMOTIONS]])
    scaled_features = scaler.transform(features_array)
    distances = kmeans.transform(scaled_features)[0]
    distances = np.where(distances == 0, 1e-9, distances)
    inverse_distances = 1 / distances
    probabilities = inverse_distances / np.sum(inverse_distances)
    
    top_3_indices = np.argsort(probabilities)[::-1][:3]
    top_3_results = []
    for idx in top_3_indices:
        top_3_results.append({
            "profile": CLUSTER_MAPPING[idx],
            "confidence_percentage": round(float(probabilities[idx]) * 100, 1)
        })
    
    return {
        "scores": scores_dict,
        "classification": {
            "primary_profile": top_3_results[0]["profile"],
            "top_3_matches": top_3_results
        }
    }

# ============== FastAPI App ==============

app = FastAPI(
    title="Datich API",
    description="Mental state analysis API using fine-tuned Qwen model",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for HF Spaces
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Datich API", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": True}

@app.post("/api/v1/analyze", response_model=AnalysisResponse)
async def analyze_text(request: AnalysisRequest):
    try:
        result = predict_mental_state(request.text)
        return AnalysisResponse(
            success=True,
            data=AnalysisData(
                scores=EmotionScores(**result["scores"]),
                classification=Classification(
                    primary_profile=result["classification"]["primary_profile"],
                    top_3_matches=[ProfileMatch(**m) for m in result["classification"]["top_3_matches"]]
                )
            ),
            message="Analysis completed successfully"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

@app.get("/api/v1/models/info")
async def get_model_info():
    return {
        "model_name": BASE_MODEL_ID,
        "adapter": "qwen_lora",
        "device": str(model.device),
        "status": "loaded"
    }
