from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes.scan import router as scan_router


app = FastAPI(
    title="CanineCue API",
    description="Dog aggression detection API",
    version="1.0.0",
)


# Allow Flutter Web to communicate with the FastAPI backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Register the scan endpoint
app.include_router(scan_router)


@app.get("/")
def root():
    return {
        "message": "CanineCue API is running"
    }