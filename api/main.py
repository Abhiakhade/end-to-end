import sys
import os

# Add project root to Python path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from fastapi import FastAPI
from src.inference.recommend import recommend

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Recommendation API running"}

@app.get("/recommend/{user_id}")
def get_recommendations(user_id: int):
    recs = recommend(user_id)
    return {"user_id": user_id, "recommendations": list(recs)}