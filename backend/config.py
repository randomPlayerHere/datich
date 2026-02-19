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