from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import analyze
import uvicorn

app = FastAPI(
    title = "Datich Backend",
    description = "API for datich",
    version= "1.0.0",
    docs_url="/docs",
    redoc_url= "/redoc"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:8080",
        "https://your-deployed-frontend.com"  # Production URL
    ],
    allow_credentials= True,
    allow_methods= ["*"],
    allow_headers= ["*"],
)
app.include_router(analyze.router,prefix="/api/v1",tags = ["Analysis"])

@app.get("/")
async def root():
    return {"message":"Datich API", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {"status":"ok"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)