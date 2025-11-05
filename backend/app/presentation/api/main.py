from fastapi import FastAPI

from app.presentation.api.v1.video_router import router as video_router


app = FastAPI(title="Vision CEX Backend", version="1.0.0")

@app.get("/")
def root():
    return {"status": "ok", "message": "Vision CEX Backend running"}

app.include_router(video_router)