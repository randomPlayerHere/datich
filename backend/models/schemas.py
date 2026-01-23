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