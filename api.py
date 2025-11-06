"""
FastAPI endpoint for Sentiment Analysis
Run with: uvicorn api:app --reload --port 8000
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import joblib
import os
import re

app = FastAPI(
    title="Sentiment Analysis API",
    description="API for sentiment analysis of text",
    version="1.0.0"
)

# Load model
def load_model():
    try:
        # Try to load model
        if os.path.exists("model.joblib"):
            model = joblib.load("model.joblib")
            print("Loaded model from model.joblib")
            return model
        print("No model found!")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

pipe = load_model()
labels = {0: "Negative", 1: "Neutral", 2: "Positive"}

def clean_text(text: str) -> str:
    """Clean text for sentiment analysis."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^A-Za-z0-9\s.,!?]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Request models
class TextRequest(BaseModel):
    text: str

class BatchRequest(BaseModel):
    texts: List[str]

class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float

class BatchSentimentResponse(BaseModel):
    results: List[dict]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: Optional[str] = None

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Sentiment Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "analyze": "/analyze - Analyze single text",
            "batch": "/batch - Analyze multiple texts",
            "health": "/health - Check API health",
            "docs": "/docs - API documentation"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and model status."""
    model_type = None
    if pipe is not None:
        # Try to get model type
        try:
            last_step = list(pipe.named_steps.values())[-1]
            model_type = type(last_step).__name__
        except:
            pass
    
    return HealthResponse(
        status="healthy" if pipe is not None else "unhealthy",
        model_loaded=pipe is not None,
        model_type=model_type
    )

@app.post("/analyze", response_model=SentimentResponse)
async def analyze_text(request: TextRequest):
    """Analyze sentiment of a single text.
    
    Args:
        request: TextRequest containing the text to analyze
    
    Returns:
        SentimentResponse with sentiment label and confidence score
    """
    if pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train a model first.")
    
    try:
        # Clean and predict
        cleaned_text = clean_text(request.text)
        probabilities = pipe.predict_proba([cleaned_text])[0]
        
        # Get prediction and confidence
        predicted_label = probabilities.argmax()
        confidence = float(probabilities.max() * 100)
        sentiment = labels[predicted_label]
        
        return SentimentResponse(
            sentiment=sentiment,
            confidence=round(confidence, 2)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing text: {str(e)}")

@app.post("/batch", response_model=BatchSentimentResponse)
async def analyze_batch(request: BatchRequest):
    """Analyze sentiment of multiple texts in batch.
    
    Args:
        request: BatchRequest containing list of texts to analyze
    
    Returns:
        BatchSentimentResponse with results for each text
    """
    if pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train a model first.")
    
    if not request.texts:
        raise HTTPException(status_code=400, detail="No texts provided")
    
    try:
        # Clean all texts
        cleaned_texts = [clean_text(text) for text in request.texts]
        
        # Predict probabilities
        all_probabilities = pipe.predict_proba(cleaned_texts)
        
        # Process results
        results = []
        for probs in all_probabilities:
            predicted_label = probs.argmax()
            confidence = float(probs.max() * 100)
            sentiment = labels[predicted_label]
            
            results.append({
                "sentiment": sentiment,
                "confidence": round(confidence, 2),
                "probabilities": {
                    "negative": round(float(probs[0] * 100), 2),
                    "neutral": round(float(probs[1] * 100), 2),
                    "positive": round(float(probs[2] * 100), 2)
                }
            })
        
        return BatchSentimentResponse(results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing batch: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

