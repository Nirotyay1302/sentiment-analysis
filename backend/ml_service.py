import os
import joblib
import numpy as np
import warnings
from textblob import TextBlob

# Try to import transformers for better sentiment analysis
TRANSFORMERS_AVAILABLE = False
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    from transformers import pipeline
    # Test if torch actually works (sometimes DLL issues on Windows)
    _ = torch.__version__
    TRANSFORMERS_AVAILABLE = True
except (ImportError, OSError, Exception) as e:
    TRANSFORMERS_AVAILABLE = False
    print(f"Transformers not available: {e}")

class TransformerSentimentModel:
    """Wrapper class for transformer-based sentiment analysis."""
    
    def __init__(self, model_name="cardiffnlp/twitter-roberta-base-sentiment-latest"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.label_map = {}
        self.label_to_num = {"Negative": 0, "Neutral": 1, "Positive": 2}
        self.num_to_label = {}
        self._load_model()
    
    def _load_model(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            if hasattr(self.model.config, 'id2label'):
                self.label_map = self.model.config.id2label
            else:
                self.label_map = {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}
            
            for idx, label_name in self.label_map.items():
                if isinstance(label_name, str):
                    label_lower = label_name.lower()
                    if "neg" in label_lower:
                        self.num_to_label[idx] = 0
                    elif "neu" in label_lower or "neutral" in label_lower:
                        self.num_to_label[idx] = 1
                    elif "pos" in label_lower:
                        self.num_to_label[idx] = 2
                    else:
                        self.num_to_label[idx] = idx
                else:
                    self.num_to_label[idx] = idx
        except Exception as e:
            raise RuntimeError(f"Failed to load transformer model: {e}")
    
    def predict(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return []
        
        try:
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_labels = predictions.argmax(dim=-1)
            
            predicted_labels = predicted_labels.cpu().numpy()
            results = [self.num_to_label.get(int(idx), 1) for idx in predicted_labels]
            return results
        except Exception:
            return [1] * len(texts)
    
    def predict_proba(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return np.array([])
        
        try:
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            probs = probabilities.cpu().numpy()
            reordered_probs = np.zeros((len(texts), 3))
            for i in range(len(texts)):
                for model_idx in range(len(probs[i])):
                    our_idx = self.num_to_label.get(model_idx, 1)
                    if 0 <= our_idx < 3:
                        reordered_probs[i][our_idx] = probs[i][model_idx]
            return reordered_probs
        except Exception:
            return np.ones((len(texts), 3)) / 3.0


class MLService:
    def __init__(self, fallback_model_path=os.path.join(os.path.dirname(__file__), "..", "model.joblib")):
        self.transformer_model = None
        self.emotion_model = None
        self.fallback_model = self._try_load_model(fallback_model_path)
        self.labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
        
        if TRANSFORMERS_AVAILABLE:
            try:
                print("Loading Transformer Model...")
                self.transformer_model = TransformerSentimentModel()
                print("Sentiment Transformer loaded successfully!")
            except Exception as e:
                print(f"Failed to load sentiment transformer: {e}")
            
            try:
                print("Loading Emotion Transformer...")
                self.emotion_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1, device=0 if torch.cuda.is_available() else -1)
                print("Emotion Transformer loaded successfully!")
            except Exception as e:
                print(f"Failed to load transformer model: {e}")

    def _try_load_model(self, path):
        try:
            if os.path.exists(path):
                return joblib.load(path)
        except Exception as e:
            print(f"Error loading fallback model from {path}: {e}")
        return None

    def analyze_sentiment(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        if self.transformer_model is not None:
            numeric_preds = self.transformer_model.predict(texts)
        elif self.fallback_model is not None:
            numeric_preds = self.fallback_model.predict(texts)
        else:
            # Final fallback to TextBlob
            numeric_preds = []
            for text in texts:
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                if polarity > 0.05:
                    numeric_preds.append(2)
                elif polarity < -0.05:
                    numeric_preds.append(0)
                else:
                    numeric_preds.append(1)
        
        # Convert to text labels
        return [self.labels.get(pred, "Neutral") for pred in numeric_preds]

    def analyze_probabilities(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        
        if self.transformer_model is not None:
            return self.transformer_model.predict_proba(texts).tolist()
        elif self.fallback_model is not None and hasattr(self.fallback_model, 'predict_proba'):
            return self.fallback_model.predict_proba(texts).tolist()
        else:
            results = []
            for text in texts:
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                if polarity > 0.05:
                    results.append([0.1, 0.2, 0.7])
                elif polarity < -0.05:
                    results.append([0.7, 0.2, 0.1])
                else:
                    results.append([0.2, 0.6, 0.2])
            return results

    def analyze_emotion(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        
        if self.emotion_model is not None:
            try:
                # Top_k=1 returns a list of lists of dicts
                results = self.emotion_model(texts)
                return [res[0]['label'].capitalize() if isinstance(res, list) else res['label'].capitalize() for res in results]
            except Exception as e:
                print(f"Emotion extraction error: {e}")
                return ["Neutral"] * len(texts)
        else:
            return ["Neutral"] * len(texts)

# Global instance
ml_service = MLService()
