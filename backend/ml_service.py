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
        # Check for custom trained model first
        custom_model_dir = os.path.join(os.path.dirname(__file__), "..", "custom_roberta_model")
        if os.path.exists(custom_model_dir) and os.path.isdir(custom_model_dir):
            # Verify it has the required files
            if os.path.exists(os.path.join(custom_model_dir, "config.json")):
                self.model_name = custom_model_dir
                print(f"Using custom fine-tuned model from: {self.model_name}")
            else:
                self.model_name = model_name
                print(f"Custom model directory incomplete, using default: {model_name}")
        else:
            self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.label_map = {}
        # Standardized label mapping: ensure consistent order [Negative, Neutral, Positive]
        self.label_to_num = {"Negative": 0, "Neutral": 1, "Positive": 2}
        self.num_to_label = {0: "Negative", 1: "Neutral", 2: "Positive"}
        self.model_label_order = None  # Track the model's actual label order
        self._load_model()
    
    def _load_model(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            if hasattr(self.model.config, 'id2label'):
                self.label_map = self.model.config.id2label
                # Build mapping from model's label indices to our standard indices
                self.model_label_order = []
                for idx in sorted(self.label_map.keys()):
                    label_name = self.label_map[idx].lower()
                    if "neg" in label_name or label_name == "0":
                        self.model_label_order.append(0)  # Negative
                    elif "neu" in label_name or label_name == "1":
                        self.model_label_order.append(1)  # Neutral
                    elif "pos" in label_name or label_name == "2":
                        self.model_label_order.append(2)  # Positive
                    else:
                        self.model_label_order.append(idx)
                print(f"Model label order: {self.model_label_order}")
            else:
                self.label_map = {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}
                self.model_label_order = [0, 1, 2]
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
            # Map model's predictions to our standard label indices
            results = []
            for idx in predicted_labels:
                if self.model_label_order:
                    # Use the model's label order mapping
                    model_idx = int(idx)
                    if 0 <= model_idx < len(self.model_label_order):
                        results.append(self.model_label_order[model_idx])
                    else:
                        results.append(1)  # Default to Neutral
                else:
                    results.append(self.num_to_label.get(int(idx), 1))
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
            # Reorder probabilities to match our standard [Negative, Neutral, Positive] order
            reordered_probs = np.zeros((len(texts), 3))
            for i in range(len(texts)):
                for model_idx in range(min(len(probs[i]), len(self.model_label_order))):
                    our_idx = self.model_label_order[model_idx] if self.model_label_order else model_idx
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
            if os.environ.get("DISABLE_HEAVY_AI") == "1":
                print("\n========================================================")
                print("⚠️  HEAVY AI DISABLED TO PREVENT CLOUD OOM CRASHES ⚠️")
                print("Falling back to extreme lightweight local scikit-learn models.")
                print("========================================================\n")
            else:
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
