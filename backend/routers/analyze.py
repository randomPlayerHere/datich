from fastapi import APIRouter, HTTPException, Depends
from models.schemas import AnalysisRequest, AnalysisResponse, SentimentScores
from models.sentiment import get_analyzer, SentimentAnalyzer

router = APIRouter()

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_text(
    request: AnalysisRequest,
    analyzer: SentimentAnalyzer = Depends(get_analyzer)
):
    try:
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