from pydantic import BaseModel, Field
from typing import Optional, List

class AnalysisRequest(BaseModel):
    text: str = Field(
        ..., 
        min_length=1, 
        max_length=5000,
        description="the text",
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Today was challenging but I managed to complete my tasks and felt accomplished."
            }
        }

class SentimentMetric(BaseModel):
    label: str = Field(..., description="Each label (e.g., 'Anxiety Indicators')")
    value: int = Field(..., ge=0, le=100, description="Score value (0-100)")

class SentimentResults(BaseModel):
    metrics: List[SentimentMetric] = Field(..., description="List of sentiment metrics")

class AnalysisResponse(BaseModel):
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