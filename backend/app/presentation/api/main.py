from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.presentation.api.v1.video_router import router as video_router, preload_default_sources
from app.presentation.api.v1.damage_router import router as damage_router
from app.presentation.api.v1.vqa_router import router as vqa_router
from app.presentation.api.v1.caption_router import router as caption_router


app = FastAPI(title="Vision CEX Backend", version="1.0.0")

# Enable permissive CORS (allow all origins). Use with caution in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def on_startup():
    # Preload sample sources so /{id}/stream works out-of-the-box
    preload_default_sources()

@app.get("/")
def root():
    return {"status": "ok", "message": "Vision CEX Backend running"}

app.include_router(video_router)
app.include_router(damage_router)
app.include_router(vqa_router)
app.include_router(caption_router)
from app.presentation.api.v1.ernie_router import router as ernie_router
app.include_router(ernie_router)