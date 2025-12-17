from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from analysis.pipeline import run_exoplanet_pipeline


app = FastAPI()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/analyze/{tic_id}")
def analyze(tic_id: int):
    return run_exoplanet_pipeline(tic_id)

@app.get("/")
def root():
    return {"status": "Backend alive"}
