# © 2025 Abishek Xavier A — All rights reserved

import os
# Configure write-friendly temporary cache directories for Vercel/serverless environments
os.environ['ASTROPY_CACHE_DIR'] = '/tmp/astropy_cache'
os.environ['ASTROPY_CONFIG_DIR'] = '/tmp/astropy_config'
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_config'

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from analysis.pipeline import run_exoplanet_pipeline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/analyze/{tic_id}")
def analyze(tic_id: int):
    return run_exoplanet_pipeline(tic_id)

@app.get("/")
def root():
    return {"status": "Backend alive"}
