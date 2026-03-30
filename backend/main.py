from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Any
from sqlalchemy.orm import Session
import datetime

from backend.ml_service import ml_service
from backend.database import get_db, PredictionHistory

app = FastAPI(
    title="Sentiment Analysis API",
    description="Backend API for Sentiment Analysis Final Year Project",
    version="1.0.0"
)

class SentimentRequest(BaseModel):
    texts: List[str]

class SentimentResponse(BaseModel):
    predictions: List[str]
    emotions: List[str]
    probabilities: List[List[float]]

@app.get("/")
def root():
    return {"message": "Welcome to the Sentiment Analysis Backend API."}

@app.post("/predict", response_model=SentimentResponse)
def predict_sentiment(request: SentimentRequest, db: Session = Depends(get_db)):
    if not request.texts:
        raise HTTPException(status_code=400, detail="Texts list cannot be empty.")
    try:
        # 1. Run inference
        predictions = ml_service.analyze_sentiment(request.texts)
        emotions = ml_service.analyze_emotion(request.texts)
        probabilities = ml_service.analyze_probabilities(request.texts)
        
        # 2. Save to database asynchronously (for a B.Tech project, sync is fine)
        for text, sentiment, emotion in zip(request.texts, predictions, emotions):
            # Let's limit text length for DB storage
            db_record = PredictionHistory(text=text[:1000], sentiment=sentiment, emotion=emotion)
            db.add(db_record)
        db.commit()
        
        return SentimentResponse(predictions=predictions, emotions=emotions, probabilities=probabilities)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
def get_history(limit: int = 100, db: Session = Depends(get_db)):
    history = db.query(PredictionHistory).order_by(PredictionHistory.timestamp.desc()).limit(limit).all()
    return [
        {
            "id": record.id,
            "text": record.text,
            "sentiment": record.sentiment,
            "emotion": record.emotion,
            "timestamp": record.timestamp
        }
        for record in history
    ]
