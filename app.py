import streamlit as st
import pandas as pd
import joblib
import os
import platform
import re
from textblob import TextBlob
import io
from threading import Thread, Event
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
from collections import Counter
import numpy as np

# Try to import sklearn metrics for accuracy calculation
try:
    from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_recall_fscore_support
    SKLEARN_METRICS_AVAILABLE = True
except ImportError:
    SKLEARN_METRICS_AVAILABLE = False
    # Fallback functions if sklearn not available
    def accuracy_score(y_true, y_pred):
        return sum(y_true == y_pred) / len(y_true) if len(y_true) > 0 else 0.0

# Try to import snscrape (optional - may not work on all platforms)
# Note: snscrape is archived and incompatible with Python 3.13+
# Completely disabled on Python 3.13+ (Streamlit Cloud uses Python 3.13)
import sys
SNSCRAPE_AVAILABLE = False
sntwitter = None

# Only attempt import on Python < 3.13 (snscrape doesn't work on 3.13+)
if sys.version_info < (3, 13):
    try:
        # Use importlib to avoid module-level import issues
        import importlib.util
        spec = importlib.util.find_spec("snscrape.modules.twitter")
        if spec is not None:
            sntwitter = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(sntwitter)  # type: ignore
            SNSCRAPE_AVAILABLE = True
    except Exception:
        # Any error means snscrape is not available
        SNSCRAPE_AVAILABLE = False
        sntwitter = None

# Try to import youtube_comment_downloader (optional)
try:
    from youtube_comment_downloader import YoutubeCommentDownloader
    YOUTUBE_DL_AVAILABLE = True
except (ImportError, Exception):
    YOUTUBE_DL_AVAILABLE = False
    YoutubeCommentDownloader = None

# Try to import transformers for better sentiment analysis
TRANSFORMERS_AVAILABLE = False
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    # Test if torch actually works (sometimes DLL issues on Windows)
    _ = torch.__version__
    TRANSFORMERS_AVAILABLE = True
except (ImportError, OSError, Exception) as e:
    TRANSFORMERS_AVAILABLE = False
    print(f"Transformers not available: {e}")

# Try to import wordcloud and matplotlib for visualization
try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

# Try to import OCR libraries for image text extraction
OCR_AVAILABLE = False
EasyOCR = None
try:
    import easyocr
    from PIL import Image
    import io
    OCR_AVAILABLE = True
    # Initialize EasyOCR reader (lazy loading - will initialize on first use)
    ocr_reader = None
except ImportError:
    OCR_AVAILABLE = False
    ocr_reader = None

# Transformer-based sentiment analysis model wrapper
class TransformerSentimentModel:
    """Wrapper class for transformer-based sentiment analysis that mimics sklearn pipeline interface."""
    
    def __init__(self, model_name="cardiffnlp/twitter-roberta-base-sentiment-latest"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.label_map = {}  # Will be populated from model config
        self.label_to_num = {"Negative": 0, "Neutral": 1, "Positive": 2}
        self._load_model()
    
    def _load_model(self):
        """Load the transformer model and tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # Get label mappings from model config
            if hasattr(self.model.config, 'id2label'):
                self.label_map = self.model.config.id2label
            else:
                # Fallback to default mapping for cardiffnlp model
                self.label_map = {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}
            
            # Create reverse mapping from label names to our numeric format
            # The model typically outputs: Negative=0, Neutral=1, Positive=2
            # But we need to check the actual label names
            self.num_to_label = {}
            for idx, label_name in self.label_map.items():
                if isinstance(label_name, str):
                    label_lower = label_name.lower()
                    if "neg" in label_lower:
                        self.num_to_label[idx] = 0  # Negative
                    elif "neu" in label_lower or "neutral" in label_lower:
                        self.num_to_label[idx] = 1  # Neutral
                    elif "pos" in label_lower:
                        self.num_to_label[idx] = 2  # Positive
                    else:
                        # Default mapping by index
                        self.num_to_label[idx] = idx
                else:
                    self.num_to_label[idx] = idx
            
        except Exception as e:
            raise RuntimeError(f"Failed to load transformer model: {e}")
    
    def predict(self, texts):
        """Predict sentiment labels for a list of texts. Returns list of integers (0=Negative, 1=Neutral, 2=Positive)."""
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return []
        
        try:
            # Tokenize inputs
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_labels = predictions.argmax(dim=-1)
            
            # Convert to numpy and map to our label format
            predicted_labels = predicted_labels.cpu().numpy()
            
            # Map transformer labels to our format (0=Negative, 1=Neutral, 2=Positive)
            results = []
            for idx in predicted_labels:
                mapped_label = self.num_to_label.get(int(idx), 1)  # Default to Neutral if mapping fails
                results.append(mapped_label)
            
            return results
        except Exception as e:
            # Fallback: return neutral predictions if error occurs
            return [1] * len(texts)
    
    def predict_proba(self, texts):
        """Predict sentiment probabilities for a list of texts. Returns numpy array of shape (n_samples, 3)."""
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return np.array([])
        
        try:
            # Tokenize inputs
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Convert to numpy and ensure correct order: [Negative, Neutral, Positive]
            probs = probabilities.cpu().numpy()
            
            # Reorder to match our label format [Negative, Neutral, Positive]
            # The model outputs probabilities for each label, we need to map them correctly
            reordered_probs = np.zeros((len(texts), 3))
            for i in range(len(texts)):
                # Map probabilities from model output indices to our format
                for model_idx in range(len(probs[i])):
                    our_idx = self.num_to_label.get(model_idx, 1)  # Default to Neutral position
                    if 0 <= our_idx < 3:
                        reordered_probs[i][our_idx] = probs[i][model_idx]
            
            return reordered_probs
        except Exception as e:
            # Fallback: return uniform probabilities if error occurs
            return np.ones((len(texts), 3)) / 3.0


# Load sentiment model for dataset/social media (resilient)
def _try_load_model(path="model.joblib"):
    try:
        return joblib.load(path)
    except Exception:
        return None

def _load_transformer_model():
    """Load transformer-based sentiment model if available."""
    if not TRANSFORMERS_AVAILABLE:
        print("WARNING: Transformers library not available. Install with: pip install transformers torch")
        return None
    
    try:
        print("Attempting to load transformer model (cardiffnlp/twitter-roberta-base-sentiment-latest)...")
        print("   This may take a few minutes on first run (~500MB download)...")
        model = TransformerSentimentModel()
        print("SUCCESS: Transformer model loaded successfully!")
        return model
    except Exception as e:
        # Don't use st.warning here as it may be called before Streamlit is initialized
        import traceback
        print(f"ERROR: Transformer model loading failed: {e}")
        print(f"   Error details: {traceback.format_exc()}")
        print("   Will try to use fallback model or TextBlob.")
        return None

# Initialize models - try transformer first, then fallback to joblib
transformer_pipe = None
pipe = _try_load_model()

# Try to load transformer model at startup if available
if TRANSFORMERS_AVAILABLE and transformer_pipe is None:
    try:
        print("Initializing transformer model at startup...")
        transformer_pipe = _load_transformer_model()
        if transformer_pipe is not None:
            print("SUCCESS: Transformer model loaded successfully at startup!")
    except Exception as e:
        print(f"WARNING: Could not load transformer model at startup: {e}")
        transformer_pipe = None

# Use transformer model if available, otherwise use joblib model
if transformer_pipe is not None:
    active_pipe = transformer_pipe
elif pipe is not None:
    active_pipe = pipe
else:
    active_pipe = None

labels = {0: "Negative", 1: "Neutral", 2: "Positive"}

def predict_sentiment(texts):
    """Predict sentiment using available model or TextBlob fallback."""
    if isinstance(texts, str):
        texts = [texts]
    
    if active_pipe is not None:
        # Use the loaded model (transformer or joblib)
        return active_pipe.predict(texts)  # type: ignore
    else:
        # Fallback to TextBlob
        results = []
        for text in texts:
            sentiment = analyze_sentiment_textblob(text)
            # Convert TextBlob labels to numeric: Negative=0, Neutral=1, Positive=2
            if sentiment == "Positive":
                results.append(2)
            elif sentiment == "Negative":
                results.append(0)
            else:
                results.append(1)
        return results

def predict_proba_sentiment(texts):
    """Predict sentiment probabilities using available model or TextBlob fallback."""
    if isinstance(texts, str):
        texts = [texts]
    
    if active_pipe is not None and hasattr(active_pipe, 'predict_proba'):
        # Use the loaded model's predict_proba
        return active_pipe.predict_proba(texts)  # type: ignore
    elif transformer_pipe is not None and hasattr(transformer_pipe, 'predict_proba'):
        return transformer_pipe.predict_proba(texts)
    else:
        # Fallback to TextBlob - estimate probabilities
        results = []
        for text in texts:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # type: ignore
            # Convert polarity (-1 to 1) to probabilities
            if polarity > 0.05:
                # Positive
                results.append([0.1, 0.2, 0.7])  # [Negative, Neutral, Positive]
            elif polarity < -0.05:
                # Negative
                results.append([0.7, 0.2, 0.1])  # [Negative, Neutral, Positive]
            else:
                # Neutral
                results.append([0.2, 0.6, 0.2])  # [Negative, Neutral, Positive]
        return np.array(results)

def ensure_model_ui():
    """In-UI helper: if no model loaded, prompt user to upload one."""
    global active_pipe, transformer_pipe, pipe
    
    # Try to load transformer model first (preferred, lazy loading)
    if transformer_pipe is None:
        # Always try to load transformer if available, even if TRANSFORMERS_AVAILABLE check failed earlier
        if TRANSFORMERS_AVAILABLE:
            try:
                with st.spinner("Loading transformer model (first time may take a moment to download ~500MB)..."):
                    transformer_pipe = _load_transformer_model()
                    if transformer_pipe is not None:
                        active_pipe = transformer_pipe
                        st.success("✅ Transformer model loaded successfully!")
                        return True
                    else:
                        st.info("⚠️ Transformer model could not be loaded. Will use TextBlob for sentiment analysis.")
            except Exception as e:
                st.warning(f"Could not load transformer model: {e}. Will use TextBlob for sentiment analysis.")
        else:
            # Try one more time to import transformers (in case it works now)
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                import torch
                with st.spinner("Loading transformer model (first time may take a moment to download ~500MB)..."):
                    transformer_pipe = _load_transformer_model()
                    if transformer_pipe is not None:
                        active_pipe = transformer_pipe
                        st.success("✅ Transformer model loaded successfully!")
                        return True
            except Exception:
                st.info("ℹ️ Transformer model not available. Using TextBlob for sentiment analysis.")
    
    # If transformer model is available, we're good
    if transformer_pipe is not None:
        active_pipe = transformer_pipe
        return True
    
    # If joblib model is available, use it
    if pipe is not None:
        active_pipe = pipe
        return True
    
    # No model available - use TextBlob as fallback and show info
    st.info("📊 Using TextBlob for sentiment analysis. For better accuracy, the transformer model will be loaded automatically when available.")
    st.caption("Note: Transformer model requires ~500MB download on first use. If you have a trained model.joblib file, you can upload it below.")
    
    # Allow uploading joblib model as fallback
    uploaded = st.file_uploader("Upload model.joblib (optional, transformer model is preferred)", type=["joblib", "pkl"], key="model_uploader")
    if uploaded is not None:
        try:
            uploaded.seek(0)
            buf = io.BytesIO(uploaded.read())
            pipe = joblib.load(buf)
            active_pipe = pipe
            st.success("Custom model loaded successfully.")
            return True
        except Exception as e:
            st.error(f"Failed to load uploaded model: {e}")
            return False
    
    # Return True even without model - we'll use TextBlob as fallback
    return True

 # Health-check server using builtin http.server so hosting platforms can probe readiness
ready_event = Event()


class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            payload = {"ready": ready_event.is_set()}
            self.wfile.write(json.dumps(payload).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


def run_health_server(port: int = 8502):
    try:
        server = HTTPServer(("0.0.0.0", port), _HealthHandler)
        server.serve_forever()
    except Exception:
        # If the platform restricts starting a TCP server, skip silently
        return


# After model and resources loaded, set readiness
ready_event.set()

# Start health server in background thread (non-blocking)
try:
    health_port = int(os.environ.get("HEALTH_PORT", "8502"))
    t = Thread(target=run_health_server, args=(health_port,), daemon=True)
    t.start()
except Exception:
    pass


def get_model_classes(model):
    """Try to retrieve class labels from the fitted model/pipeline."""
    try:
        # common: pipeline with classifier named 'clf' or last step
        if hasattr(model, 'classes_'):
            return list(model.classes_)
        if hasattr(model, 'named_steps'):
            # take last estimator
            last = list(model.named_steps.values())[-1]
            if hasattr(last, 'classes_'):
                return list(last.classes_)
        # fallback: unknown
    except Exception:
        pass
    return None


def is_numerical_column(series, threshold=0.8):
    """
    Check if a column is primarily numerical.
    
    Args:
        series: pandas Series to check
        threshold: Minimum proportion of numerical values to consider column as numerical
    
    Returns:
        True if column is primarily numerical, False otherwise
    """
    if len(series) == 0:
        return False
    
    # Convert to string and check
    series_str = series.astype(str)
    
    # Count how many values are purely numerical (digits only, possibly with decimals)
    numeric_count = 0
    total_count = 0
    
    for val in series_str:
        val_str = str(val).strip()
        if val_str and val_str.lower() not in ['nan', 'none', 'null', '']:
            total_count += 1
            # Check if value is purely numerical (digits, decimal point, minus sign)
            if re.match(r'^-?\d+\.?\d*$', val_str):
                numeric_count += 1
    
    if total_count == 0:
        return False
    
    # If more than threshold% are numerical, consider it a numerical column
    return (numeric_count / total_count) >= threshold

def detect_text_column(df, exclude_numerical=True):
    """
    Detect the best text column in a dataframe, excluding numerical columns.
    
    Args:
        df: pandas DataFrame
        exclude_numerical: If True, exclude columns that are primarily numerical
    
    Returns:
        Name of the best text column, or None if not found
    """
    cols = list(df.columns)
    detected_col = None
    
    # First, try to find column with 'text' in name
    for c in cols:
        if 'text' in str(c).lower() or 'comment' in str(c).lower() or 'review' in str(c).lower():
            if not exclude_numerical or not is_numerical_column(df[c]):
                detected_col = c
                break
    
    # If not found, score columns by average string length (excluding numerical columns)
    if detected_col is None:
        scores = {}
        for c in cols:
            # Skip if numerical column
            if exclude_numerical and is_numerical_column(df[c]):
                continue
            
            try:
                vals = df[c].astype(str).replace('nan', '').tolist()
                # Calculate average length of non-empty, non-numerical values
                lengths = []
                for v in vals:
                    v_str = str(v).strip()
                    if v_str and v_str.lower() not in ['nan', 'none', 'null', '']:
                        # Skip if purely numerical
                        if not re.match(r'^-?\d+\.?\d*$', v_str):
                            lengths.append(len(re.sub(r'[^A-Za-z0-9\s]', '', v_str).strip()))
                
                if lengths:
                    scores[c] = sum(lengths) / len(lengths)
                else:
                    scores[c] = 0
            except Exception:
                scores[c] = 0
        
        if scores:
            detected_col = max(scores, key=scores.get)  # type: ignore
    
    return detected_col

def read_csv_with_header_detection(uploaded):
    """Read a CSV and try to detect if the real header is on a later row (common with exported CSVs that include a title row).

    Returns (df, header_row_index_or_None)
    """
    try:
        # Create a text buffer for preview
        if hasattr(uploaded, 'read'):
            uploaded.seek(0)
            sample_text = uploaded.read().decode('utf-8', errors='ignore')
            preview_buf = io.StringIO(sample_text)
        else:
            preview_buf = uploaded

        preview = pd.read_csv(preview_buf, header=None, nrows=10, dtype=str, keep_default_na=False)
        header_row = None
        for i, row in preview.iterrows():
            row_vals = ' '.join([str(x).lower() for x in row.tolist()])
            if re.search(r'\btext\b', row_vals):
                header_row = i
                break

        # Fallback: if first row contains title-like text, assume header is second row
        if header_row is None:
            first0 = str(preview.iloc[0, 0]).lower()
            if 'social media' in first0 or 'sentiments' in first0 or 'social' in first0:
                header_row = 1

        # Read the full CSV using detected header
        if hasattr(uploaded, 'read'):
            uploaded.seek(0)
            if header_row is not None:
                df = pd.read_csv(uploaded, header=header_row, dtype=str, keep_default_na=False)  # type: ignore
            else:
                uploaded.seek(0)
                df = pd.read_csv(uploaded, header=0, dtype=str, keep_default_na=False)
        else:
            if header_row is not None:
                df = pd.read_csv(uploaded, header=header_row, dtype=str, keep_default_na=False)  # type: ignore
            else:
                df = pd.read_csv(uploaded, header=0, dtype=str, keep_default_na=False)

        return df, header_row
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return None, None

st.set_page_config(page_title="Social Media Sentiment Analyzer", layout="centered", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    body {background-color: #f5f6fa;}
    .main {background-color: #fff; border-radius: 12px; padding: 2rem;}
    .stButton>button {background-color: #0066cc; color: white; border-radius: 6px;}
    .stFileUploader {border-radius: 6px;}
    .stTextInput>div>div>input {border-radius: 6px;}
    .stTextArea>div>textarea {border-radius: 6px;}
    .stDataFrame {background-color: #f9f9f9; border-radius: 6px;}
    .stAlert {border-radius: 6px;}
    .st-bb {background: #0066cc !important; color: white !important;}
    .stApp {padding-bottom: 60px;}
    .footer {position: fixed; left: 0; bottom: 0; width: 100%; background: #f5f6fa; color: #888; text-align: center; padding: 10px 0; font-size: 0.9rem;}
    .branding {display: flex; align-items: center; gap: 1rem; margin-bottom: 1.5rem;}
    .branding img {height: 48px;}
    .branding-title {font-size: 2.1rem; font-weight: 700; color: #0066cc; letter-spacing: 1px;}
    </style>
    """,
    unsafe_allow_html=True
)



st.sidebar.header("Choose Mode")
mode = st.sidebar.selectbox(
    "Select Analysis Mode",
    [
        "Analyze Dataset",
        "Analyze Social Media Link",
        "Analyze Image/Screenshot",
        "Compare Two Datasets",
        "Accuracy Meter/Validation",
        "Manual Text Input"
    ],
    key="mode_selectbox"
)

# Add Model Info section
st.sidebar.markdown("---")
st.sidebar.header("📊 Model Info")
if transformer_pipe is not None:
    st.sidebar.success("✅ Transformer Model Active")
    st.sidebar.caption("Model: RoBERTa-base<br>Fine-tuned for sentiment<br>High accuracy", unsafe_allow_html=True)
    st.sidebar.info("💡 **Using state-of-the-art transformer model!**")
elif active_pipe is not None:
    st.sidebar.success("✅ Model Loaded")
    st.sidebar.caption("Model type: Custom trained model<br>Features: TF-IDF + N-grams", unsafe_allow_html=True)
    st.sidebar.info("💡 **Tip:** Transformer model available for better accuracy!")
else:
    st.sidebar.info("📊 Using TextBlob")
    st.sidebar.caption("Transformer model will load automatically<br>or upload 'model.joblib'", unsafe_allow_html=True)

# Add OCR Info section
st.sidebar.markdown("---")
st.sidebar.header("📸 OCR Status")
if OCR_AVAILABLE:
    st.sidebar.success("✅ OCR Available")
    st.sidebar.caption("Image text extraction<br>enabled with EasyOCR", unsafe_allow_html=True)
else:
    st.sidebar.info("ℹ️ OCR Not Available")
    st.sidebar.caption("Install: pip install easyocr Pillow")

st.markdown("<h1 style='text-align: center; color: #0066cc; margin-bottom: 0.5em;'>📊 Social Media Sentiment Analyzer</h1>", unsafe_allow_html=True)

def is_numeric_only(text):
    """Check if text contains only numbers and no meaningful text."""
    if not text or pd.isna(text):
        return True
    
    text_str = str(text).strip()
    if not text_str:
        return True
    
    # Remove whitespace and common punctuation
    cleaned = re.sub(r'[\s.,!?\-_()\[\]{}]', '', text_str)
    
    # Check if it's all digits or mostly digits
    if cleaned.isdigit():
        return True
    
    # Check if it's mostly numbers (more than 80% digits)
    if len(cleaned) > 0:
        digit_ratio = sum(c.isdigit() for c in cleaned) / len(cleaned)
        if digit_ratio > 0.8:
            return True
    
    # Check if it's a very short string with mostly numbers
    if len(text_str) <= 3 and any(c.isdigit() for c in text_str):
        if sum(c.isdigit() for c in text_str) >= len(text_str) * 0.7:
            return True
    
    return False

def clean_text(text):
    """Clean and preprocess text data, filtering out numeric-only content."""
    if pd.isna(text) or text == "":
        return ""
    
    text = str(text)
    # Skip if text is purely numeric
    if re.match(r'^[\d\s.,+-]+$', text.strip()):
        return ""
    
def is_numeric_column(series, threshold=0.8):
    """
    Check if a column is primarily numeric.
    
    Args:
        series: pandas Series to check
        threshold: Minimum proportion of numeric values to consider column as numeric
    
    Returns:
        True if column is primarily numeric, False otherwise
    """
    if len(series) == 0:
        return False
    
    # Convert to string and check
    series_str = series.astype(str)
    
    # Count numeric-like values (pure numbers, or numbers with minimal text)
    numeric_count = 0
    total_count = 0
    
    for val in series_str:
        val_str = str(val).strip()
        if val_str == '' or pd.isna(val):
            continue
        
        total_count += 1
        # Check if value is primarily numeric
        # Remove common separators and check if remaining is mostly digits
        cleaned = re.sub(r'[,\s$€£¥%]', '', val_str)
        if cleaned == '':
            continue
        
        # Check if it's a pure number or mostly numeric
        if re.match(r'^[\d.+-]+$', cleaned) or (len(re.findall(r'\d', cleaned)) / max(len(cleaned), 1)) > 0.7:
            numeric_count += 1
    
    if total_count == 0:
        return False
    
    # If more than threshold% are numeric, consider it a numeric column
    return (numeric_count / total_count) > threshold

def filter_text_columns(df):
    """
    Filter out numeric columns and return only text columns suitable for sentiment analysis.
    
    Args:
        df: pandas DataFrame
    
    Returns:
        List of column names that are suitable for text analysis
    """
    text_columns = []
    for col in df.columns:
        # Skip if column is numeric type
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        
        # Check if column contains primarily text (not numeric)
        if not is_numeric_column(df[col]):
            text_columns.append(col)
    
    return text_columns

def analyze_sentiment_textblob(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity  # type: ignore
    if polarity > 0.05:
        return "Positive"
    elif polarity < -0.05:
        return "Negative"
    else:
        return "Neutral"

def analyze_sentiment_textblob(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity  # type: ignore
    if polarity > 0.05:
        return "Positive"
    elif polarity < -0.05:
        return "Negative"
    else:
        return "Neutral"


def render_pie_chart(sentiment_counts, title="Sentiment Distribution", colors=None):
    """Render a pie chart. Try Plotly, fall back to Matplotlib, else show a table.

    sentiment_counts: pandas Series (index=labels, values=counts)
    colors: list of colors aligned with sentiment_counts.index
    """
    try:
        import plotly.graph_objects as go

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=sentiment_counts.index.tolist(),
                    values=sentiment_counts.values.tolist(),
                    hole=0.3,
                    marker=dict(colors=colors if colors is not None else ["#2ecc71", "#f1c40f", "#e74c3c"]),
                )
            ]
        )
        fig.update_layout(title=title, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        return
    except Exception:
        # Plotly not available or error occurred - try matplotlib
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            ax.pie(
                sentiment_counts.values.tolist(),
                labels=sentiment_counts.index.tolist(),
                autopct="%1.1f%%",
                colors=colors if colors is not None else ["#2ecc71", "#f1c40f", "#e74c3c"],
            )
            ax.axis("equal")
            plt.title(title)
            st.pyplot(fig)
            return
        except Exception:
            # Fallback: textual summary
            st.write(f"**{title}**")
            st.write(sentiment_counts)
            return

def fetch_twitter_replies(url, limit=100):
    """Fetch Twitter replies - requires snscrape library."""
    if not SNSCRAPE_AVAILABLE:
        return []
    
    tweet_id = url.split("/")[-1]
    replies = []
    try:
        for i, tweet in enumerate(sntwitter.TwitterSearchScraper(f'conversation_id:{tweet_id}').get_items()):  # type: ignore
            if i >= limit:
                break
            if tweet.inReplyToTweetId == int(tweet_id):  # type: ignore
                replies.append(tweet.content)  # type: ignore
    except Exception as e:
        pass
    return replies

def fetch_youtube_comments(url, limit=100):
    """Fetch YouTube comments - requires youtube-comment-downloader library."""
    if not YOUTUBE_DL_AVAILABLE:
        return []
    
    comments = []
    try:
        downloader = YoutubeCommentDownloader()  # type: ignore
        for i, comment in enumerate(downloader.get_comments_from_url(url, sort_by=0)):  # type: ignore
            if i >= limit:
                break
            comments.append(comment["text"])
    except Exception as e:
        pass
    return comments

def generate_wordcloud(text_list, title="Word Cloud"):
    """Generate and display a word cloud from list of texts."""
    if not WORDCLOUD_AVAILABLE:
        st.info("Word cloud visualization not available. Install wordcloud library.")
        return
    
    if not text_list:
        st.warning("No text data available for word cloud.")
        return
    
    try:
        # Combine all texts
        text = ' '.join(str(t) for t in text_list if t and str(t).strip())
        
        if not text.strip():
            st.warning("No valid text data for word cloud generation.")
            return
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            max_words=100,
            colormap='viridis'
        ).generate(text)
        
        # Display
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, fontsize=16, pad=20)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error generating word cloud: {str(e)}")

def plot_timeseries(df, date_col, sentiment_col):
    """Plot sentiment over time if date column exists."""
    try:
        # Try to convert to datetime
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        # Group by date and sentiment
        df_with_date = df[df[date_col].notna()].copy()
        if df_with_date.empty:
            return False
        
        df_with_date = df_with_date.groupby([df_with_date[date_col].dt.date, sentiment_col]).size().unstack(fill_value=0)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        df_with_date.plot(kind='line', ax=ax, marker='o')
        ax.set_title('Sentiment Over Time', fontsize=14)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.legend(title='Sentiment')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        return True
    except Exception:
        return False

def get_top_keywords(texts, sentiment_label, n=10):
    """Extract top keywords from texts for a specific sentiment."""
    # Combine all texts
    all_text = ' '.join(str(t).lower() for t in texts if t)
    
    # Remove common stopwords
    stopwords = set(['the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at'])
    words = re.findall(r'\b[a-z]{3,}\b', all_text.lower())
    words = [w for w in words if w not in stopwords]
    
    # Count and return top N
    counter = Counter(words)
    return counter.most_common(n)

def calculate_accuracy_metrics(y_true, y_pred, label_names=["Negative", "Neutral", "Positive"]):
    """
    Calculate comprehensive accuracy metrics.
    
    Args:
        y_true: List of actual labels (0, 1, 2 or "Negative", "Neutral", "Positive")
        y_pred: List of predicted labels (0, 1, 2 or "Negative", "Neutral", "Positive")
        label_names: List of label names
    
    Returns:
        Dictionary with metrics
    """
    # Convert labels to numeric if needed
    label_map = {"Negative": 0, "Neutral": 1, "Positive": 2}
    
    y_true_num = []
    y_pred_num = []
    
    for label in y_true:
        if isinstance(label, str):
            y_true_num.append(label_map.get(label, 1))
        else:
            y_true_num.append(int(label))
    
    for label in y_pred:
        if isinstance(label, str):
            y_pred_num.append(label_map.get(label, 1))
        else:
            y_pred_num.append(int(label))
    
    y_true_num = np.array(y_true_num)
    y_pred_num = np.array(y_pred_num)
    
    # Calculate overall accuracy
    if SKLEARN_METRICS_AVAILABLE:
        acc = accuracy_score(y_true_num, y_pred_num)
    else:
        acc = sum(y_true_num == y_pred_num) / len(y_true_num) if len(y_true_num) > 0 else 0.0
    
    metrics = {
        "accuracy": float(acc),
        "total_samples": len(y_true_num),
        "correct_predictions": int(np.sum(y_true_num == y_pred_num)),
        "incorrect_predictions": int(np.sum(y_true_num != y_pred_num))
    }
    
    # Calculate per-class metrics if sklearn is available
    if SKLEARN_METRICS_AVAILABLE:
        try:
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true_num, y_pred_num, labels=[0, 1, 2], average=None, zero_division=0
            )
            
            # Per-class metrics
            class_metrics = {}
            for i, label_name in enumerate(label_names):
                class_metrics[label_name] = {
                    "precision": float(precision[i]),
                    "recall": float(recall[i]),
                    "f1_score": float(f1[i]),
                    "support": int(support[i])
                }
            metrics["per_class"] = class_metrics
            
            # Macro averages
            metrics["macro_avg"] = {
                "precision": float(np.mean(precision)),
                "recall": float(np.mean(recall)),
                "f1_score": float(np.mean(f1))
            }
            
            # Confusion matrix
            cm = confusion_matrix(y_true_num, y_pred_num, labels=[0, 1, 2])
            metrics["confusion_matrix"] = cm.tolist()
            
            # Classification report text
            try:
                report = classification_report(y_true_num, y_pred_num, target_names=label_names, output_dict=False)
                metrics["classification_report"] = report
            except:
                metrics["classification_report"] = "Unable to generate classification report"
        except Exception as e:
            metrics["error"] = f"Error calculating detailed metrics: {e}"
    else:
        # Basic per-class accuracy without sklearn
        class_metrics = {}
        for i, label_name in enumerate(label_names):
            mask = y_true_num == i
            if np.sum(mask) > 0:
                class_acc = np.sum((y_true_num == i) & (y_pred_num == i)) / np.sum(mask)
                class_metrics[label_name] = {"accuracy": float(class_acc), "support": int(np.sum(mask))}
            else:
                class_metrics[label_name] = {"accuracy": 0.0, "support": 0}
        metrics["per_class"] = class_metrics
    
    return metrics

def plot_confusion_matrix(cm, labels=["Negative", "Neutral", "Positive"]):
    """Plot confusion matrix."""
    try:
        import matplotlib.pyplot as plt
        try:
            import seaborn as sns
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix')
            plt.tight_layout()
            return fig
        except ImportError:
            # Fallback to matplotlib only
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
            ax.figure.colorbar(im, ax=ax)
            ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
                   xticklabels=labels, yticklabels=labels,
                   title='Confusion Matrix', ylabel='Actual', xlabel='Predicted')
            
            # Add text annotations
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], 'd'),
                           ha="center", va="center",
                           color="white" if cm[i, j] > thresh else "black")
            plt.tight_layout()
            return fig
    except:
        return None

def initialize_ocr_reader():
    """Initialize EasyOCR reader (lazy loading to avoid slow startup)."""
    global ocr_reader
    if not OCR_AVAILABLE:
        return None
    if ocr_reader is None:
        try:
            # Initialize EasyOCR reader for English (GPU disabled for cloud compatibility)
            # This will download models on first use (~100MB, takes 30-60 seconds)
            ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            return ocr_reader
        except Exception as e:
            print(f"Failed to initialize OCR reader: {e}")
            return None
    return ocr_reader

def extract_text_from_image(image_file):
    """
    Extract text from an image using OCR.
    
    Args:
        image_file: Uploaded image file (Streamlit UploadedFile)
    
    Returns:
        List of extracted text strings
    """
    if not OCR_AVAILABLE:
        return []
    
    try:
        # Initialize OCR reader
        reader = initialize_ocr_reader()
        if reader is None:
            return []
        
        # Read image
        image = Image.open(image_file)
        
        # Convert to numpy array for EasyOCR (numpy already imported at top)
        img_array = np.array(image)
        
        # Perform OCR
        results = reader.readtext(img_array)
        
        # Extract text from results
        extracted_texts = []
        for (bbox, text, confidence) in results:
            # Filter out low confidence results
            if confidence > 0.3:  # Adjust threshold as needed
                extracted_texts.append(text)
        
        return extracted_texts
    except Exception as e:
        st.error(f"Error extracting text from image: {e}")
        return []

# ----------- Mode 1: Dataset Analyzer -----------
if mode == "Analyze Dataset":
    st.subheader("📂 Batch Dataset Sentiment Analysis")
    st.markdown("""
    Upload a CSV file containing a column with text data. The app will perform a quick scan of up to 100 rows by default (fast), and you can opt to analyze the full dataset.
    """)
    # Ensure model is available before analysis
    if not ensure_model_ui():
        st.stop()

    uploaded_file = st.file_uploader("Upload a dataset (CSV)", type=["csv"], key="dataset_uploader")
    if uploaded_file is not None:
        df, header_row = read_csv_with_header_detection(uploaded_file)
        if df is None:
            st.error("Could not parse uploaded CSV.")
        else:
            if header_row is not None:
                st.info(f"Detected header row at line {header_row + 1} (0-indexed: {header_row}). If this is wrong, override below.")  # type: ignore
            # Allow user to override the header row if detection fails
            override_header = st.checkbox("Override detected header row / specify header row manually", value=False, key="override_header_checkbox")
            if override_header:
                hdr = st.number_input("Header row index (0-based)", min_value=0, max_value=1000, value=header_row if header_row is not None else 0, step=1, key="header_row_input")  # type: ignore
                try:
                    # Re-read uploaded file with user-specified header
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, header=int(hdr), dtype=str, keep_default_na=False)
                    st.success(f"Re-read CSV with header row = {hdr}")
                except Exception as e:
                    st.error(f"Failed to re-read CSV with header={hdr}: {e}")

        if df is not None:
            # Let the user pick which column contains the text to analyze
            # Use improved detection that excludes numerical columns
            cols = list(df.columns)
            detected_col = detect_text_column(df, exclude_numerical=True)
            
            # Filter out numerical columns from selection
            non_numerical_cols = [c for c in cols if not is_numerical_column(df[c])]
            
            if not non_numerical_cols:
                st.error("⚠️ No suitable text columns found. All columns appear to be numerical. Please ensure your dataset contains text data.")
                st.stop()
            
            # Warn about numerical columns
            numerical_cols = [c for c in cols if is_numerical_column(df[c])]
            if numerical_cols:
                st.info(f"ℹ️ Excluded {len(numerical_cols)} numerical column(s) from text analysis: {', '.join(numerical_cols[:5])}")
            
            # Select text column (default to detected, or first non-numerical)
            if detected_col and detected_col in non_numerical_cols:
                default_idx = non_numerical_cols.index(detected_col)
            else:
                default_idx = 0
            
            text_col = st.selectbox(
                "Select text column for sentiment analysis (numerical columns excluded)",
                options=non_numerical_cols,
                index=default_idx,
                key="dataset_text_col"
            )
            
            # Validate selected column
            sample_vals = df[text_col].astype(str).head(20).tolist()
            num_numeric = sum(1 for v in sample_vals if re.fullmatch(r"^-?\d+\.?\d*$", str(v).strip()))
            if num_numeric > 5:
                st.warning(f"⚠️ The selected column '{text_col}' appears to contain many numerical values. Please verify this is the correct text column.")
            total_rows = len(df)
            st.write(f"Rows in dataset: **{total_rows}** — using column: **{text_col}**")

            # Quick scan sample size (user-configurable)
            default_sample = min(100, total_rows)
            max_sample = min(10000, total_rows)
            sample_size = st.number_input(
                "Quick scan sample size (rows)",
                min_value=1,
                max_value=max_sample,
                value=default_sample,
                step=1,
                key="sample_size_input",
            )
            st.info(f"Quick scan will analyze {sample_size} row(s). Toggle below to analyze the full dataset.")

            analyze_full = st.checkbox("Analyze full dataset (may take longer)", value=False, key="analyze_full_checkbox")

            if analyze_full:
                texts = df[text_col].astype(str).apply(clean_text).tolist()
                
                # Filter out numeric-only and empty texts
                valid_indices = []
                filtered_count = 0
                for i, text in enumerate(texts):
                    if text.strip() and not is_numeric_only(text):
                        valid_indices.append(i)
                    else:
                        filtered_count += 1
                
                if filtered_count > 0:
                    st.info(f"ℹ️ Filtered out {filtered_count} numeric-only or empty entries (not suitable for sentiment analysis)")
                
                texts = [texts[i] for i in valid_indices]
                df_filtered = df.iloc[valid_indices].copy() if valid_indices else df.copy()
                
                # Ensure model UI is set up (will try to load transformer)
                ensure_model_ui()
                
                if len(texts) == 0:
                    st.warning("⚠️ No valid text entries found after filtering. Please check your dataset.")
                    st.stop()
                
                # Add progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Process in chunks with progress updates
                chunk_size = max(1, len(texts) // 50)  # Update 50 times
                preds = []
                for i in range(0, len(texts), chunk_size):
                    chunk = texts[i:i+chunk_size]
                    chunk_preds = predict_sentiment(chunk)
                    preds.extend(chunk_preds)
                    
                    # Update progress
                    progress = min(1.0, (i + len(chunk)) / len(texts))
                    progress_bar.progress(progress)
                    status_text.text(f"Processed {len(preds)}/{len(texts)} texts...")
                
                progress_bar.empty()
                status_text.empty()
                
                df_filtered["Predicted"] = preds
                df_filtered["Predicted_Label"] = df_filtered["Predicted"].map(labels)  # type: ignore
                result_df = df_filtered
                st.success(f"Full dataset analysis complete - {len(texts)} valid entries analyzed")
            else:
                # Ensure model UI is set up (will try to load transformer)
                ensure_model_ui()
                
                # Use first N rows (deterministic) instead of random sampling
                head_df = df.head(int(sample_size)).copy()
                head_texts = head_df[text_col].astype(str).apply(clean_text).tolist()
                
                # Filter out numeric-only and empty texts
                valid_indices = []
                filtered_count = 0
                for i, text in enumerate(head_texts):
                    if text.strip() and not is_numeric_only(text):
                        valid_indices.append(i)
                    else:
                        filtered_count += 1
                
                if filtered_count > 0:
                    st.info(f"ℹ️ Filtered out {filtered_count} numeric-only or empty entries (not suitable for sentiment analysis)")
                
                head_texts = [head_texts[i] for i in valid_indices]
                head_df_filtered = head_df.iloc[valid_indices].copy() if valid_indices else head_df.copy()
                
                if len(head_texts) == 0:
                    st.warning("⚠️ No valid text entries found after filtering. Please check your dataset.")
                    st.stop()
                
                head_preds = predict_sentiment(head_texts)
                head_df_filtered["Predicted"] = head_preds
                head_df_filtered["Predicted_Label"] = head_df_filtered["Predicted"].map(labels)  # type: ignore
                result_df = head_df_filtered
                st.success(f"Quick scan (first {sample_size} rows) complete - {len(head_texts)} valid entries analyzed")

            # Allow filtering by sentiment (based on text predictions)
            sentiment_choices = ["Positive", "Neutral", "Negative"]
            selected_sentiments = st.multiselect("Filter results by sentiment (text) — leave empty to show all", options=sentiment_choices, default=sentiment_choices, key="dataset_sentiment_filter")

            # Show sample of results
            st.markdown("**Sample results:**")
            filtered_df = result_df[result_df["Predicted_Label"].isin(selected_sentiments)] if selected_sentiments else result_df
            st.dataframe(filtered_df.head(20))

            # Sentiment distribution (pie + bar)
            # Normalize label order to Positive, Neutral, Negative for consistent colors
            ordered_labels = ["Positive", "Neutral", "Negative"]
            sentiment_counts = filtered_df["Predicted_Label"].value_counts().reindex(ordered_labels, fill_value=0) if not filtered_df.empty else pd.Series([0,0,0], index=ordered_labels)  # type: ignore
            if sentiment_counts.sum() > 0:
                render_pie_chart(sentiment_counts, colors=["#2ecc71", "#f1c40f", "#e74c3c"])
            else:
                st.info("No rows match the selected sentiment filters.")

            # Bar chart for counts
            st.markdown("**Counts (bar):**")
            st.bar_chart(sentiment_counts)

            # Advanced Visualizations
            st.markdown("---")
            st.markdown("## 📊 Advanced Analytics")
            
            # Word Clouds by sentiment
            if WORDCLOUD_AVAILABLE and not filtered_df.empty:
                st.markdown("### 💭 Word Clouds by Sentiment")
                try:
                    pos_texts = filtered_df[filtered_df["Predicted_Label"] == "Positive"][text_col].tolist() if len(filtered_df[filtered_df["Predicted_Label"] == "Positive"]) > 0 else []
                    neg_texts = filtered_df[filtered_df["Predicted_Label"] == "Negative"][text_col].tolist() if len(filtered_df[filtered_df["Predicted_Label"] == "Negative"]) > 0 else []
                    
                    if pos_texts:
                        st.markdown("**Positive Sentiment Words:**")
                        generate_wordcloud(pos_texts, "Positive Sentiment Word Cloud")
                    
                    if neg_texts:
                        st.markdown("**Negative Sentiment Words:**")
                        generate_wordcloud(neg_texts, "Negative Sentiment Word Cloud")
                except Exception as e:
                    st.info("Could not generate word clouds. Need more data.")
            
            # Top Keywords
            st.markdown("### 🔑 Top Keywords by Sentiment")
            try:
                for sent_label in ["Positive", "Negative"]:
                    sent_texts = filtered_df[filtered_df["Predicted_Label"] == sent_label][text_col].tolist()
                    if sent_texts:
                        keywords = get_top_keywords(sent_texts, sent_label, n=10)
                        if keywords:
                            st.write(f"**{sent_label}:**")
                            keyword_str = ", ".join([f"{word} ({count})" for word, count in keywords])
                            st.write(keyword_str)
            except Exception:
                pass
            
            # Time-series analysis if date column exists
            st.markdown("### 📈 Sentiment Over Time")
            date_columns = [col for col in filtered_df.columns if any(x in col.lower() for x in ['date', 'time', 'timestamp'])]
            if date_columns:
                st.info(f"Found potential date columns: {', '.join(date_columns)}")
                date_col_selected = st.selectbox("Select date column for time-series analysis:", options=date_columns + ['None'], key="date_col_select")
                if date_col_selected and date_col_selected != 'None':
                    if plot_timeseries(filtered_df, date_col_selected, "Predicted_Label"):
                        st.success("Time-series chart generated!")
            else:
                st.info("No date/timestamp columns found. Time-series analysis not available.")
            
            st.markdown("---")
            
            # Show top positive / negative examples (if available)
            st.markdown("## 📝 Top Examples")
            try:
                pos_examples = result_df[result_df["Predicted_Label"] == "Positive"][text_col].head(5)  # type: ignore
                neg_examples = result_df[result_df["Predicted_Label"] == "Negative"][text_col].head(5)  # type: ignore
                st.write("**Top Positive examples:**")
                for i, v in enumerate(pos_examples, 1):
                    st.write(f"{i}. {v}")
                st.write("**Top Negative examples:**")
                for i, v in enumerate(neg_examples, 1):
                    st.write(f"{i}. {v}")
            except Exception:
                pass

            # Download results (filtered view)
            try:
                download_df = filtered_df if not filtered_df.empty else result_df
                csv_bytes = download_df.to_csv(index=False).encode("utf-8")
                st.download_button("Download results as CSV", data=csv_bytes, file_name="sentiment_results.csv", mime="text/csv", key="download_results")
            except Exception as e:
                st.error(f"Could not prepare download: {e}")

# ----------- Mode 2: Social Media Analyzer (Twitter/YouTube) -----------
elif mode == "Analyze Social Media Link":
    st.subheader("📱 Social Media Sentiment Analysis")
    
    # Informative message about social media scraping
    st.info("""
    **ℹ️ Note**: Social media scraping has limitations on cloud platforms due to API restrictions and library compatibility.
    
    **Alternative Options**:
    - 📊 **Upload CSV**: Export social media data and upload as CSV (Recommended)
    - 📝 **Manual Input**: Copy and paste text manually
    - 🔗 **Use Dataset Mode**: Best for bulk analysis
    """)
    
    # Show status of scraping libraries
    col1, col2 = st.columns(2)
    with col1:
        if SNSCRAPE_AVAILABLE:
            st.success("✅ Twitter scraping available")
        else:
            st.info("ℹ️ Twitter scraping not available (use CSV upload instead)")
    
    with col2:
        if YOUTUBE_DL_AVAILABLE:
            st.success("✅ YouTube scraping available")
        else:
            st.info("ℹ️ YouTube scraping not available (use CSV upload instead)")
    
    # Provide option to upload CSV as alternative
    st.markdown("---")
    st.markdown("### Option 1: Upload Social Media Data (CSV)")
    st.markdown("Export your social media data (Twitter/YouTube comments) to CSV and upload here:")
    uploaded_csv = st.file_uploader("Upload social media data (CSV)", type=["csv"], key="social_csv_uploader")
    
    if uploaded_csv is not None:
        st.info("💡 Switch to 'Analyze Dataset' mode to analyze the uploaded CSV file.")
        st.markdown("---")
    
    # Original link-based approach
    st.markdown("### Option 2: Direct Link Analysis")
    st.markdown("*Note: This feature may not work on all platforms due to API restrictions.*")
    
    link = st.text_input("Paste a Twitter or YouTube link (optional):")
    if st.button("Fetch & Analyze"):
        # Ensure model is available before analysis
        if not ensure_model_ui():
            st.stop()

        comments = []

        # Twitter support
        if "twitter.com" in link or "x.com" in link:
            if not SNSCRAPE_AVAILABLE:
                st.error("""
                **Twitter scraping is not available on this platform.**
                
                **Recommended alternatives**:
                1. **Export Twitter data**: Use Twitter's export feature or third-party tools
                2. **Upload as CSV**: Switch to 'Analyze Dataset' mode and upload your CSV
                3. **Manual input**: Copy and paste tweets manually in 'Manual Text Input' mode
                
                *Note: Twitter API restrictions and library compatibility issues prevent direct scraping on cloud platforms.*
                """)
                st.stop()
            comments = fetch_twitter_replies(link, limit=100)

        # YouTube support
        elif "youtube.com" in link or "youtu.be" in link:
            if not YOUTUBE_DL_AVAILABLE:
                st.error("""
                **YouTube comment scraping is not available on this platform.**
                
                **Recommended alternatives**:
                1. **Export comments**: Use YouTube Data API or browser extensions to export comments
                2. **Upload as CSV**: Switch to 'Analyze Dataset' mode and upload your CSV
                3. **Manual input**: Copy and paste comments manually in 'Manual Text Input' mode
                
                *Note: YouTube API restrictions prevent direct comment scraping on cloud platforms.*
                """)
                st.stop()
            comments = fetch_youtube_comments(link, limit=100)
        
        elif not link:
            st.warning("Please provide a valid Twitter or YouTube link, or upload a CSV file above.")
            st.stop()
        
        else:
            st.error("Invalid link. Please provide a valid Twitter (twitter.com or x.com) or YouTube (youtube.com) link.")
            st.stop()

        if comments:
            cleaned_comments = [clean_text(c) for c in comments]
            
            # Ensure model UI is set up (will try to load transformer)
            ensure_model_ui()
            
            # Add progress bar for social media analysis
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process with progress updates
            chunk_size = max(1, len(cleaned_comments) // 20)  # Update 20 times
            preds = []
            for i in range(0, len(cleaned_comments), chunk_size):
                chunk = cleaned_comments[i:i+chunk_size]
                chunk_preds = predict_sentiment(chunk)
                preds.extend(chunk_preds)
                
                # Update progress
                progress = min(1.0, (i + len(chunk)) / len(cleaned_comments))
                progress_bar.progress(progress)
                status_text.text(f"Analyzing {len(preds)}/{len(cleaned_comments)} comments...")
            
            progress_bar.empty()
            status_text.empty()
            
            df = pd.DataFrame({"Comment": comments, "Predicted": preds})
            df["Sentiment"] = df["Predicted"].map(labels)  # type: ignore
            # Sentiment filter
            sentiment_choices = ["Positive", "Neutral", "Negative"]
            selected_sentiments = st.multiselect("Filter comments by sentiment (text)", options=sentiment_choices, default=sentiment_choices, key="social_sentiment_filter")

            filtered_comments = df[df["Sentiment"].isin(selected_sentiments)] if selected_sentiments else df

            st.dataframe(filtered_comments.head(20))
            ordered_labels = ["Positive", "Neutral", "Negative"]
            counts = filtered_comments["Sentiment"].value_counts().reindex(ordered_labels, fill_value=0)  # type: ignore
            if counts.sum() > 0:
                render_pie_chart(counts, colors=["#2ecc71", "#f1c40f", "#e74c3c"])
                st.markdown("**Counts (bar):**")
                st.bar_chart(counts)
            else:
                st.info("No comments match the selected sentiment filters.")

            # Download filtered comments
            try:
                csv_bytes = filtered_comments.to_csv(index=False).encode("utf-8")
                st.download_button("Download comments as CSV", data=csv_bytes, file_name="social_comments.csv", mime="text/csv", key="download_social")
            except Exception as e:
                st.error(f"Could not prepare download: {e}")
        else:
            st.error("No comments found or could not fetch comments.")

# ----------- Mode 3: Image/Screenshot Analyzer -----------
elif mode == "Analyze Image/Screenshot":
    st.subheader("📸 Image & Screenshot Sentiment Analysis")
    
    # Check OCR availability
    if not OCR_AVAILABLE:
        st.error("""
        **OCR (Optical Character Recognition) is not available.**
        
        **To enable OCR**:
        1. Install required packages: `pip install easyocr Pillow`
        2. Restart the app
        
        **Alternative**: Use 'Manual Text Input' mode to copy and paste text from images.
        """)
        st.stop()
    
    st.info("""
    **📸 Upload an image or screenshot** containing text (comments, reviews, social media posts, etc.)
    
    **Supported formats**: PNG, JPG, JPEG, WEBP
    **Best results**: Clear, high-contrast images with readable text
    """)
    
    # Upload image
    uploaded_image = st.file_uploader(
        "Upload an image or screenshot",
        type=["png", "jpg", "jpeg", "webp"],
        key="image_uploader"
    )
    
    if uploaded_image is not None:
        # Display the uploaded image
        st.markdown("### 📷 Uploaded Image")
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Extract text button
        if st.button("Extract Text & Analyze Sentiment", key="extract_analyze_btn"):
            # Ensure model is available
            if not ensure_model_ui():
                st.stop()
            
            # Extract text from image
            with st.spinner("Extracting text from image using OCR (this may take 30-60 seconds on first use to download models)..."):
                # Reset file pointer
                uploaded_image.seek(0)
                extracted_texts = extract_text_from_image(uploaded_image)
            
            if not extracted_texts:
                st.warning("⚠️ No text could be extracted from the image. Please try:")
                st.markdown("""
                - Ensure the image is clear and text is readable
                - Try a higher resolution image
                - Check that the image contains visible text
                - Use 'Manual Text Input' mode to type the text manually
                """)
            else:
                # Display extracted text
                st.markdown("### 📝 Extracted Text")
                full_text = " ".join(extracted_texts)
                
                # Show individual text segments
                with st.expander("View extracted text segments"):
                    for i, text in enumerate(extracted_texts, 1):
                        st.write(f"**Segment {i}**: {text}")
                
                # Show full combined text
                st.text_area("Full extracted text:", value=full_text, height=150, key="extracted_text_display")
                
                # Analyze sentiment
                if full_text.strip():
                    st.markdown("### 📊 Sentiment Analysis Results")
                    
                    # Clean text
                    cleaned_text = clean_text(full_text)
                    
                    if not cleaned_text.strip():
                        st.warning("No valid text found after cleaning. Cannot analyze sentiment.")
                    else:
                        # Get predictions
                        if active_pipe is not None:
                            # Use model for prediction
                            prediction = active_pipe.predict([cleaned_text])[0]  # type: ignore
                            
                            # Get probabilities if available
                            if hasattr(active_pipe, 'predict_proba'):
                                probas = active_pipe.predict_proba([cleaned_text])[0]  # type: ignore
                            elif transformer_pipe is not None:
                                probas = transformer_pipe.predict_proba([cleaned_text])[0]
                            else:
                                probas = np.array([0.33, 0.33, 0.34])
                            
                            sentiment_label = labels[prediction]  # type: ignore
                            confidence = probas.max() * 100
                            
                            # Display results
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Sentiment", sentiment_label)
                            with col2:
                                st.metric("Confidence", f"{confidence:.1f}%")
                            with col3:
                                # Color code based on sentiment
                                if sentiment_label == "Positive":
                                    st.success("✅ Positive")
                                elif sentiment_label == "Negative":
                                    st.error("❌ Negative")
                                else:
                                    st.info("➖ Neutral")
                            
                            # Probability breakdown
                            st.markdown("#### Probability Breakdown")
                            prob_df = pd.DataFrame({
                                "Sentiment": ["Negative", "Neutral", "Positive"],
                                "Probability": [f"{probas[0]*100:.1f}%", f"{probas[1]*100:.1f}%", f"{probas[2]*100:.1f}%"]
                            })
                            st.dataframe(prob_df, use_container_width=True, hide_index=True)
                            
                            # Visual indicator
                            if confidence > 80:
                                st.success(f"✅ High confidence prediction ({confidence:.1f}%)")
                            elif confidence > 60:
                                st.info(f"⚠️ Moderate confidence prediction ({confidence:.1f}%)")
                            else:
                                st.warning(f"⚠️ Low confidence prediction ({confidence:.1f}%) - results may be unreliable")
                            
                            # Analyze individual segments if multiple
                            if len(extracted_texts) > 1:
                                st.markdown("---")
                                st.markdown("### 📊 Individual Segment Analysis")
                                
                                segment_results = []
                                for i, text_segment in enumerate(extracted_texts, 1):
                                    cleaned_segment = clean_text(text_segment)
                                    if cleaned_segment.strip():
                                        seg_pred = active_pipe.predict([cleaned_segment])[0]  # type: ignore
                                        seg_label = labels[seg_pred]  # type: ignore
                                        segment_results.append({
                                            "Segment": i,
                                            "Text": text_segment[:100] + "..." if len(text_segment) > 100 else text_segment,
                                            "Sentiment": seg_label
                                        })
                                
                                if segment_results:
                                    segment_df = pd.DataFrame(segment_results)
                                    st.dataframe(segment_df, use_container_width=True, hide_index=True)
                                    
                                    # Segment sentiment distribution
                                    st.markdown("#### Segment Sentiment Distribution")
                                    seg_counts = segment_df["Sentiment"].value_counts()
                                    if not seg_counts.empty:
                                        st.bar_chart(seg_counts)
                        else:
                            # Fallback to TextBlob
                            st.warning("Model not available. Using TextBlob for basic sentiment analysis.")
                            sentiment = analyze_sentiment_textblob(cleaned_text)
                            st.metric("Sentiment", sentiment)
                
                # Download results (only if analysis was performed)
                if full_text.strip() and active_pipe is not None:
                    st.markdown("---")
                    st.markdown("### 💾 Download Results")
                    try:
                        results_data = {
                            "extracted_text": [full_text],
                            "sentiment": [sentiment_label],
                            "confidence": [f"{confidence:.1f}%"]
                        }
                        results_df = pd.DataFrame(results_data)
                        csv_bytes = results_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "Download results as CSV",
                            data=csv_bytes,
                            file_name="image_sentiment_results.csv",
                            mime="text/csv",
                            key="download_image_results"
                        )
                    except NameError:
                        # Variables not defined (analysis didn't complete)
                        pass

# ----------- Mode 4: Compare Two Datasets -----------
elif mode == "Compare Two Datasets":
    st.subheader("🔄 Compare Two Datasets")
    
    st.info("""
    **Compare a labeled dataset (with declared sentiments) with an unlabeled dataset (non-declared).**
    
    **How it works**:
    1. Upload **Dataset 1**: Reference dataset with actual sentiment labels (declared)
    2. Upload **Dataset 2**: Test dataset without labels (non-declared)
    3. Match datasets (by row index or text similarity)
    4. Predict sentiments for Dataset 2
    5. Compare predictions with Dataset 1 labels
    6. View accuracy metrics
    """)
    
    # Ensure model is available
    if not ensure_model_ui():
        st.stop()
    
    st.markdown("---")
    st.markdown("### Step 1: Upload Reference Dataset (With Labels)")
    st.markdown("Upload a CSV file with text and **actual sentiment labels** (Positive, Negative, Neutral):")
    
    reference_file = st.file_uploader(
        "Upload reference dataset with labels (CSV)",
        type=["csv"],
        key="compare_reference_uploader"
    )
    
    st.markdown("---")
    st.markdown("### Step 2: Upload Test Dataset (Without Labels)")
    st.markdown("Upload a CSV file with text data **without sentiment labels**:")
    
    test_file = st.file_uploader(
        "Upload test dataset without labels (CSV)",
        type=["csv"],
        key="compare_test_uploader"
    )
    
    if reference_file is not None and test_file is not None:
        # Read both datasets
        df_ref, header_row_ref = read_csv_with_header_detection(reference_file)
        df_test, header_row_test = read_csv_with_header_detection(test_file)
        
        if df_ref is None or df_test is None:
            st.error("Could not parse one or both CSV files.")
        else:
            st.success(f"✅ Reference dataset: {len(df_ref)} rows | Test dataset: {len(df_test)} rows")
            
            # Detect columns for reference dataset
            ref_cols = list(df_ref.columns)
            ref_text_col = detect_text_column(df_ref, exclude_numerical=True)
            ref_sentiment_col = None
            for c in ref_cols:
                if 'sentiment' in str(c).lower() or 'label' in str(c).lower():
                    ref_sentiment_col = c
                    break
            
            # Detect columns for test dataset
            test_cols = list(df_test.columns)
            test_text_col = detect_text_column(df_test, exclude_numerical=True)
            
            # Filter numerical columns
            ref_non_num_cols = [c for c in ref_cols if not is_numerical_column(df_ref[c])]
            test_non_num_cols = [c for c in test_cols if not is_numerical_column(df_test[c])]
            
            if not ref_non_num_cols or not test_non_num_cols:
                st.error("⚠️ No suitable text columns found in one or both datasets. Please ensure datasets contain text data.")
                st.stop()
            
            st.markdown("---")
            st.markdown("### Step 3: Select Columns")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Reference Dataset (With Labels):**")
                ref_text_col = st.selectbox(
                    "Text column:",
                    options=ref_non_num_cols,
                    index=ref_non_num_cols.index(ref_text_col) if ref_text_col and ref_text_col in ref_non_num_cols else 0,
                    key="compare_ref_text_col"
                )
                if ref_sentiment_col:
                    ref_sentiment_col = st.selectbox(
                        "Sentiment/Label column:",
                        options=ref_cols,
                        index=ref_cols.index(ref_sentiment_col),
                        key="compare_ref_sentiment_col"
                    )
                else:
                    ref_sentiment_col = st.selectbox(
                        "Sentiment/Label column:",
                        options=ref_cols,
                        key="compare_ref_sentiment_col"
                    )
            
            with col2:
                st.markdown("**Test Dataset (Without Labels):**")
                test_text_col = st.selectbox(
                    "Text column:",
                    options=test_non_num_cols,
                    index=test_non_num_cols.index(test_text_col) if test_text_col and test_text_col in test_non_num_cols else 0,
                    key="compare_test_text_col"
                )
            
            # Show previews
            st.markdown("---")
            st.markdown("### Preview")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Reference Dataset:**")
                st.dataframe(df_ref[[ref_text_col, ref_sentiment_col]].head(5))
            
            with col2:
                st.markdown("**Test Dataset:**")
                st.dataframe(df_test[[test_text_col]].head(5))
            
            # Matching method
            st.markdown("---")
            st.markdown("### Step 4: Select Matching Method")
            match_method = st.radio(
                "How to match datasets:",
                ["By Row Index (same order)", "By Text Similarity (match similar texts)"],
                key="match_method"
            )
            
            if st.button("Compare Datasets & Calculate Accuracy", key="compare_datasets_btn"):
                # Prepare data
                ref_texts = df_ref[ref_text_col].astype(str).apply(clean_text).tolist()
                ref_labels = df_ref[ref_sentiment_col].astype(str).str.strip().tolist()
                
                # Normalize labels
                label_normalize = {
                    "positive": "Positive", "negative": "Negative", "neutral": "Neutral",
                    "POSITIVE": "Positive", "NEGATIVE": "Negative", "NEUTRAL": "Neutral"
                }
                ref_labels = [label_normalize.get(l, l) for l in ref_labels]
                
                test_texts = df_test[test_text_col].astype(str).apply(clean_text).tolist()
                
                # Match datasets
                matched_ref_texts = []
                matched_ref_labels = []
                matched_test_texts = []
                matched_indices = []
                
                if match_method == "By Row Index (same order)":
                    # Match by index (assume same order)
                    min_len = min(len(ref_texts), len(test_texts))
                    matched_ref_texts = ref_texts[:min_len]
                    matched_ref_labels = ref_labels[:min_len]
                    matched_test_texts = test_texts[:min_len]
                    matched_indices = list(range(min_len))
                    
                    if len(ref_texts) != len(test_texts):
                        st.warning(f"⚠️ Datasets have different lengths. Comparing first {min_len} rows.")
                else:
                    # Match by text similarity
                    with st.spinner("Matching texts by similarity..."):
                        from difflib import SequenceMatcher
                        
                        for i, test_text in enumerate(test_texts):
                            best_match_idx = -1
                            best_similarity = 0.0
                            
                            for j, ref_text in enumerate(ref_texts):
                                similarity = SequenceMatcher(None, test_text.lower(), ref_text.lower()).ratio()
                                if similarity > best_similarity and similarity > 0.7:  # 70% similarity threshold
                                    best_similarity = similarity
                                    best_match_idx = j
                            
                            if best_match_idx >= 0:
                                matched_test_texts.append(test_text)
                                matched_ref_texts.append(ref_texts[best_match_idx])
                                matched_ref_labels.append(ref_labels[best_match_idx])
                                matched_indices.append((i, best_match_idx))
                        
                        if not matched_test_texts:
                            st.error("❌ No matching texts found (similarity threshold: 70%). Try 'By Row Index' method instead.")
                            st.stop()
                        else:
                            st.info(f"✅ Matched {len(matched_test_texts)} texts (similarity threshold: 70%)")
                
                if not matched_test_texts:
                    st.error("No matched texts found. Please check your datasets.")
                    st.stop()
                
                # Predict sentiments for test dataset
                with st.spinner("Predicting sentiments for test dataset..."):
                    if active_pipe is not None:
                        predicted_numeric = active_pipe.predict(matched_test_texts)  # type: ignore
                        predicted_labels = [labels[p] for p in predicted_numeric]  # type: ignore
                    else:
                        st.error("Model not available.")
                        st.stop()
                
                # Calculate metrics
                with st.spinner("Calculating accuracy metrics..."):
                    metrics = calculate_accuracy_metrics(matched_ref_labels, predicted_labels)
                
                # Display results
                st.markdown("---")
                st.markdown("## 📊 Comparison Results")
                
                # Summary
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Overall Accuracy", f"{metrics['accuracy']*100:.2f}%")
                with col2:
                    st.metric("Matched Samples", len(matched_test_texts))
                with col3:
                    st.metric("Correct", metrics['correct_predictions'])
                with col4:
                    st.metric("Incorrect", metrics['incorrect_predictions'], delta_color="inverse")
                
                # Accuracy assessment
                accuracy_percent = metrics['accuracy'] * 100
                if accuracy_percent >= 80:
                    st.success(f"✅ Excellent accuracy: {accuracy_percent:.2f}%")
                elif accuracy_percent >= 60:
                    st.info(f"⚠️ Good accuracy: {accuracy_percent:.2f}%")
                else:
                    st.warning(f"⚠️ Low accuracy: {accuracy_percent:.2f}% - Consider improving the model")
                
                # Per-class metrics
                if "per_class" in metrics:
                    st.markdown("### Per-Class Metrics")
                    class_data = []
                    for label, class_metrics in metrics["per_class"].items():
                        if "precision" in class_metrics:
                            class_data.append({
                                "Class": label,
                                "Precision": f"{class_metrics['precision']*100:.2f}%",
                                "Recall": f"{class_metrics['recall']*100:.2f}%",
                                "F1-Score": f"{class_metrics['f1_score']*100:.2f}%",
                                "Support": class_metrics['support']
                            })
                    if class_data:
                        class_df = pd.DataFrame(class_data)
                        st.dataframe(class_df, use_container_width=True, hide_index=True)
                
                # Confusion Matrix
                if "confusion_matrix" in metrics:
                    st.markdown("### Confusion Matrix")
                    cm = np.array(metrics["confusion_matrix"])
                    cm_fig = plot_confusion_matrix(cm)
                    if cm_fig:
                        st.pyplot(cm_fig)
                
                # Detailed comparison
                st.markdown("### Detailed Comparison")
                comparison_df = pd.DataFrame({
                    "Test Text": [t[:80] + "..." if len(t) > 80 else t for t in matched_test_texts],
                    "Reference Label": matched_ref_labels,
                    "Predicted Label": predicted_labels,
                    "Match": ["✅" if a == p else "❌" for a, p in zip(matched_ref_labels, predicted_labels)]
                })
                
                # Filter
                match_filter = st.selectbox("Filter:", ["All", "Correct", "Incorrect"], key="compare_filter")
                if match_filter == "Correct":
                    comparison_df = comparison_df[comparison_df["Match"] == "✅"]
                elif match_filter == "Incorrect":
                    comparison_df = comparison_df[comparison_df["Match"] == "❌"]
                
                st.dataframe(comparison_df.head(50), use_container_width=True, hide_index=True)
                
                # Download results
                st.markdown("---")
                st.markdown("### 💾 Download Results")
                results_df = pd.DataFrame({
                    "test_text": matched_test_texts,
                    "reference_label": matched_ref_labels,
                    "predicted_label": predicted_labels,
                    "match": [a == p for a, p in zip(matched_ref_labels, predicted_labels)]
                })
                csv_bytes = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download comparison results as CSV",
                    data=csv_bytes,
                    file_name="dataset_comparison_results.csv",
                    mime="text/csv",
                    key="download_comparison_results"
                )

# ----------- Mode 5: Accuracy Meter/Validation -----------
elif mode == "Accuracy Meter/Validation":
    st.subheader("📊 Accuracy Meter & Model Validation")
    
    st.info("""
    **Compare predicted sentiments with actual labels to measure model accuracy.**
    
    **Two comparison modes**:
    1. **Single Dataset**: Compare predictions with labels in the same dataset
    2. **Two Datasets**: Compare a reference dataset (with labels) with a test dataset (without labels)
    """)
    
    # Ensure model is available
    if not ensure_model_ui():
        st.stop()
    
    # Comparison mode selection
    comparison_mode = st.radio(
        "Select comparison mode:",
        ["Single Dataset (with labels)", "Two Datasets (reference + test)"],
        key="comparison_mode_radio"
    )
    
    if comparison_mode == "Single Dataset (with labels)":
        st.markdown("### Step 1: Upload Reference Dataset")
        st.markdown("Upload a CSV file with text and actual sentiment labels:")
        
        reference_file = st.file_uploader(
            "Upload reference dataset (CSV with text and sentiment columns)",
            type=["csv"],
            key="accuracy_reference_uploader"
        )
    else:
        st.markdown("### Step 1: Upload Reference Dataset (with labels)")
        st.markdown("Upload a CSV file with text and actual sentiment labels:")
        
        reference_file = st.file_uploader(
            "Upload reference dataset (CSV with text and sentiment columns)",
            type=["csv"],
            key="accuracy_reference_uploader_2"
        )
        
        st.markdown("### Step 2: Upload Test Dataset (without labels)")
        st.markdown("Upload a CSV file with text only (no sentiment labels):")
        
        test_file = st.file_uploader(
            "Upload test dataset (CSV with text column only)",
            type=["csv"],
            key="accuracy_test_uploader"
        )
    
    if reference_file is not None:
        # Read reference dataset
        df_ref, header_row = read_csv_with_header_detection(reference_file)
        
        if df_ref is None:
            st.error("Could not parse uploaded CSV.")
        else:
            st.success(f"✅ Reference dataset loaded: {len(df_ref)} rows")
            
            # Auto-detect columns (excluding numerical)
            cols = list(df_ref.columns)
            text_col = detect_text_column(df_ref, exclude_numerical=True)
            
            # Find sentiment/label column
            sentiment_col = None
            for c in cols:
                if 'sentiment' in str(c).lower() or 'label' in str(c).lower():
                    sentiment_col = c
                    break
            
            # Filter numerical columns
            non_numerical_cols = [c for c in cols if not is_numerical_column(df_ref[c])]
            
            if not non_numerical_cols:
                st.error("⚠️ No suitable text columns found. All columns appear to be numerical.")
                st.stop()
            
            # Warn about numerical columns
            numerical_cols = [c for c in cols if is_numerical_column(df_ref[c])]
            if numerical_cols:
                st.info(f"ℹ️ Excluded {len(numerical_cols)} numerical column(s) from text analysis: {', '.join(numerical_cols[:5])}")
            
            # Let user select columns
            st.markdown("### Step 2: Select Columns")
            if text_col and text_col in non_numerical_cols:
                default_idx = non_numerical_cols.index(text_col)
            else:
                default_idx = 0
            
            text_col = st.selectbox(
                "Select text column (numerical columns excluded):",
                options=non_numerical_cols,
                index=default_idx,
                key="accuracy_text_col"
            )
            
            if sentiment_col:
                sentiment_col = st.selectbox("Select sentiment/label column:", options=cols, index=cols.index(sentiment_col), key="accuracy_sentiment_col")
            else:
                sentiment_col = st.selectbox("Select sentiment/label column:", options=cols, key="accuracy_sentiment_col")
            
            # Show preview
            st.markdown("### Preview")
            preview_df = df_ref[[text_col, sentiment_col]].head(10)
            st.dataframe(preview_df)
            
            # Validate labels
            unique_labels = df_ref[sentiment_col].unique()
            valid_labels = ["Positive", "Negative", "Neutral", "positive", "negative", "neutral", "POSITIVE", "NEGATIVE", "NEUTRAL"]
            
            invalid_labels = [l for l in unique_labels if str(l).strip() not in valid_labels]
            if invalid_labels:
                st.warning(f"⚠️ Found invalid labels: {invalid_labels}. Expected: Positive, Negative, Neutral")
            
            if st.button("Calculate Accuracy", key="calculate_accuracy_btn"):
                # Clean and prepare data
                texts = df_ref[text_col].astype(str).apply(clean_text).tolist()
                actual_labels = df_ref[sentiment_col].astype(str).str.strip().tolist()
                
                # Normalize labels
                label_normalize = {
                    "positive": "Positive",
                    "negative": "Negative",
                    "neutral": "Neutral",
                    "POSITIVE": "Positive",
                    "NEGATIVE": "Negative",
                    "NEUTRAL": "Neutral"
                }
                actual_labels = [label_normalize.get(l, l) for l in actual_labels]
                
                # Predict sentiments
                with st.spinner("Predicting sentiments..."):
                    if active_pipe is not None:
                        predicted_numeric = active_pipe.predict(texts)  # type: ignore
                        predicted_labels = [labels[p] for p in predicted_numeric]  # type: ignore
                    else:
                        st.error("Model not available. Please ensure a model is loaded.")
                        st.stop()
                
                # Calculate metrics
                with st.spinner("Calculating accuracy metrics..."):
                    metrics = calculate_accuracy_metrics(actual_labels, predicted_labels)
                
                # Display results
                st.markdown("---")
                st.markdown("## 📊 Accuracy Results")
                
                # Overall metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Overall Accuracy", f"{metrics['accuracy']*100:.2f}%")
                with col2:
                    st.metric("Total Samples", metrics['total_samples'])
                with col3:
                    st.metric("Correct", metrics['correct_predictions'], delta=f"+{metrics['correct_predictions']}")
                with col4:
                    st.metric("Incorrect", metrics['incorrect_predictions'], delta=f"-{metrics['incorrect_predictions']}", delta_color="inverse")
                
                # Accuracy visualization
                accuracy_percent = metrics['accuracy'] * 100
                if accuracy_percent >= 80:
                    st.success(f"✅ Excellent accuracy: {accuracy_percent:.2f}%")
                elif accuracy_percent >= 60:
                    st.info(f"⚠️ Good accuracy: {accuracy_percent:.2f}%")
                else:
                    st.warning(f"⚠️ Low accuracy: {accuracy_percent:.2f}% - Consider retraining the model")
                
                # Per-class metrics
                if "per_class" in metrics:
                    st.markdown("### Per-Class Metrics")
                    class_data = []
                    for label, class_metrics in metrics["per_class"].items():
                        if "precision" in class_metrics:
                            class_data.append({
                                "Class": label,
                                "Precision": f"{class_metrics['precision']*100:.2f}%",
                                "Recall": f"{class_metrics['recall']*100:.2f}%",
                                "F1-Score": f"{class_metrics['f1_score']*100:.2f}%",
                                "Support": class_metrics['support']
                            })
                        else:
                            class_data.append({
                                "Class": label,
                                "Accuracy": f"{class_metrics['accuracy']*100:.2f}%",
                                "Support": class_metrics['support']
                            })
                    
                    if class_data:
                        class_df = pd.DataFrame(class_data)
                        st.dataframe(class_df, use_container_width=True, hide_index=True)
                
                # Macro averages
                if "macro_avg" in metrics:
                    st.markdown("### Macro Averages")
                    macro_col1, macro_col2, macro_col3 = st.columns(3)
                    with macro_col1:
                        st.metric("Macro Precision", f"{metrics['macro_avg']['precision']*100:.2f}%")
                    with macro_col2:
                        st.metric("Macro Recall", f"{metrics['macro_avg']['recall']*100:.2f}%")
                    with macro_col3:
                        st.metric("Macro F1-Score", f"{metrics['macro_avg']['f1_score']*100:.2f}%")
                
                # Confusion Matrix
                if "confusion_matrix" in metrics:
                    st.markdown("### Confusion Matrix")
                    cm = np.array(metrics["confusion_matrix"])
                    cm_fig = plot_confusion_matrix(cm)
                    if cm_fig:
                        st.pyplot(cm_fig)
                    else:
                        # Fallback: show as table
                        cm_df = pd.DataFrame(
                            cm,
                            index=["Actual: Negative", "Actual: Neutral", "Actual: Positive"],
                            columns=["Pred: Negative", "Pred: Neutral", "Pred: Positive"]
                        )
                        st.dataframe(cm_df)
                
                # Classification Report
                if "classification_report" in metrics and isinstance(metrics["classification_report"], str):
                    st.markdown("### Classification Report")
                    st.text(metrics["classification_report"])
                
                # Detailed comparison table
                st.markdown("### Detailed Comparison")
                comparison_df = pd.DataFrame({
                    "Text": [t[:100] + "..." if len(t) > 100 else t for t in texts],
                    "Actual": actual_labels,
                    "Predicted": predicted_labels,
                    "Match": ["✅" if a == p else "❌" for a, p in zip(actual_labels, predicted_labels)]
                })
                
                # Filter options
                match_filter = st.selectbox("Filter by match status:", ["All", "Correct", "Incorrect"], key="match_filter")
                if match_filter == "Correct":
                    comparison_df = comparison_df[comparison_df["Match"] == "✅"]
                elif match_filter == "Incorrect":
                    comparison_df = comparison_df[comparison_df["Match"] == "❌"]
                
                st.dataframe(comparison_df.head(50), use_container_width=True, hide_index=True)
                
                # Download results
                st.markdown("---")
                st.markdown("### 💾 Download Results")
                results_df = pd.DataFrame({
                    "text": texts,
                    "actual_label": actual_labels,
                    "predicted_label": predicted_labels,
                    "match": [a == p for a, p in zip(actual_labels, predicted_labels)]
                })
                csv_bytes = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download accuracy results as CSV",
                    data=csv_bytes,
                    file_name="accuracy_results.csv",
                    mime="text/csv",
                    key="download_accuracy_results"
                )

# ----------- Mode 5: Compare Two Datasets -----------
elif mode == "Compare Two Datasets":
    st.subheader("📊 Compare Two Datasets")
    
    st.info("""
    **Compare two different datasets to check accuracy and consistency.**
    
    **How it works**:
    1. Upload **Dataset 1**: Reference dataset with declared/known labels (Positive, Negative, Neutral)
    2. Upload **Dataset 2**: Test dataset without labels (will be predicted)
    3. The app will predict sentiments for Dataset 2 (skipping numeric-only entries)
    4. Compare predictions with Dataset 1 labels (if matching texts found)
    5. View accuracy metrics and detailed comparison
    """)
    
    # Ensure model is available
    if not ensure_model_ui():
        st.stop()
    
    st.markdown("### Step 1: Upload Reference Dataset (With Labels)")
    st.markdown("Upload a CSV file with text and actual sentiment labels:")
    
    reference_file = st.file_uploader(
        "Upload reference dataset with labels (CSV)",
        type=["csv"],
        key="compare_ref_uploader"
    )
    
    st.markdown("### Step 2: Upload Test Dataset (Without Labels)")
    st.markdown("Upload a CSV file with text only (labels will be predicted, numeric-only entries will be skipped):")
    
    test_file = st.file_uploader(
        "Upload test dataset without labels (CSV)",
        type=["csv"],
        key="compare_test_uploader"
    )
    
    if reference_file is not None and test_file is not None:
        # Read both datasets
        df_ref, header_row_ref = read_csv_with_header_detection(reference_file)
        df_test, header_row_test = read_csv_with_header_detection(test_file)
        
        if df_ref is None or df_test is None:
            st.error("Could not parse one or both CSV files.")
        else:
            st.success(f"✅ Reference dataset: {len(df_ref)} rows | Test dataset: {len(df_test)} rows")
            
            # Auto-detect columns for reference
            ref_cols = list(df_ref.columns)
            ref_text_col = None
            ref_sentiment_col = None
            
            for c in ref_cols:
                if 'text' in str(c).lower() or 'comment' in str(c).lower():
                    ref_text_col = c
                if 'sentiment' in str(c).lower() or 'label' in str(c).lower():
                    ref_sentiment_col = c
            
            if ref_text_col is None:
                for c in ref_cols:
                    if df_ref[c].dtype == 'object':
                        ref_text_col = c
                        break
            
            # Auto-detect columns for test
            test_cols = list(df_test.columns)
            test_text_col = None
            
            for c in test_cols:
                if 'text' in str(c).lower() or 'comment' in str(c).lower():
                    test_text_col = c
                    break
            
            if test_text_col is None:
                for c in test_cols:
                    if df_test[c].dtype == 'object':
                        test_text_col = c
                        break
            
            # Let user select columns
            st.markdown("### Step 3: Select Columns")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Reference Dataset:**")
                ref_text_col = st.selectbox("Text column:", options=ref_cols, index=ref_cols.index(ref_text_col) if ref_text_col else 0, key="ref_text_col")
                if ref_sentiment_col:
                    ref_sentiment_col = st.selectbox("Sentiment column:", options=ref_cols, index=ref_cols.index(ref_sentiment_col), key="ref_sentiment_col")
                else:
                    ref_sentiment_col = st.selectbox("Sentiment column:", options=ref_cols, key="ref_sentiment_col")
            
            with col2:
                st.markdown("**Test Dataset:**")
                test_text_col = st.selectbox("Text column:", options=test_cols, index=test_cols.index(test_text_col) if test_text_col else 0, key="test_text_col")
            
            # Show previews
            st.markdown("### Preview")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Reference Dataset (first 5 rows):**")
                st.dataframe(df_ref[[ref_text_col, ref_sentiment_col]].head(5))
            with col2:
                st.markdown("**Test Dataset (first 5 rows):**")
                st.dataframe(df_test[[test_text_col]].head(5))
            
            if st.button("Compare Datasets & Calculate Accuracy", key="compare_datasets_btn"):
                # Process reference dataset - filter numeric
                ref_texts_raw = df_ref[ref_text_col].astype(str).tolist()
                ref_labels_raw = df_ref[ref_sentiment_col].astype(str).str.strip().tolist()
                
                ref_texts = []
                ref_labels = []
                for text, label in zip(ref_texts_raw, ref_labels_raw):
                    cleaned = clean_text(text)
                    if cleaned and not is_numeric_only(text):
                        ref_texts.append(cleaned)
                        ref_labels.append(label)
                
                # Normalize reference labels
                label_normalize = {
                    "positive": "Positive", "negative": "Negative", "neutral": "Neutral",
                    "POSITIVE": "Positive", "NEGATIVE": "Negative", "NEUTRAL": "Neutral"
                }
                ref_labels = [label_normalize.get(l, l) for l in ref_labels]
                
                # Process test dataset - filter out numeric-only text
                test_texts_raw = df_test[test_text_col].astype(str).tolist()
                test_texts = []
                test_indices = []
                
                with st.spinner("Filtering numeric-only text and preparing data..."):
                    for idx, text in enumerate(test_texts_raw):
                        cleaned = clean_text(text)
                        if cleaned and not is_numeric_only(text):
                            test_texts.append(cleaned)
                            test_indices.append(idx)
                
                skipped_count = len(test_texts_raw) - len(test_texts)
                if skipped_count > 0:
                    st.info(f"ℹ️ Skipped {skipped_count} numeric-only or empty entries from test dataset")
                
                if not test_texts:
                    st.error("No valid text found in test dataset after filtering. Please check your data.")
                    st.stop()
                
                # Predict sentiments for test dataset
                with st.spinner(f"Predicting sentiments for {len(test_texts)} texts..."):
                    if active_pipe is not None:
                        predicted_numeric = active_pipe.predict(test_texts)  # type: ignore
                        predicted_labels = [labels[p] for p in predicted_numeric]  # type: ignore
                    else:
                        st.error("Model not available. Please ensure a model is loaded.")
                        st.stop()
                
                # Create mapping from reference texts to labels
                ref_text_to_label = {}
                for text, label in zip(ref_texts, ref_labels):
                    ref_text_to_label[text.lower().strip()] = label
                
                # Match test predictions with reference labels
                matched_predictions = []
                matched_actuals = []
                matched_texts_list = []
                unmatched_predictions = []
                unmatched_texts = []
                
                for test_text, pred_label in zip(test_texts, predicted_labels):
                    test_key = test_text.lower().strip()
                    if test_key in ref_text_to_label:
                        matched_predictions.append(pred_label)
                        matched_actuals.append(ref_text_to_label[test_key])
                        matched_texts_list.append(test_text)
                    else:
                        unmatched_predictions.append(pred_label)
                        unmatched_texts.append(test_text)
                
                # Display results
                st.markdown("---")
                st.markdown("## 📊 Comparison Results")
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Test Samples", len(test_texts))
                with col2:
                    st.metric("Matched with Reference", len(matched_predictions))
                with col3:
                    st.metric("Unmatched (New)", len(unmatched_predictions))
                with col4:
                    st.metric("Skipped (Numeric)", skipped_count)
                
                # Calculate accuracy for matched samples
                if matched_predictions and matched_actuals:
                    st.markdown("### Accuracy for Matched Samples")
                    
                    metrics = calculate_accuracy_metrics(matched_actuals, matched_predictions)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
                    with col2:
                        st.metric("Correct", metrics['correct_predictions'])
                    with col3:
                        st.metric("Incorrect", metrics['incorrect_predictions'], delta_color="inverse")
                    with col4:
                        st.metric("Total Matched", metrics['total_samples'])
                    
                    # Accuracy visualization
                    accuracy_percent = metrics['accuracy'] * 100
                    if accuracy_percent >= 80:
                        st.success(f"✅ Excellent accuracy: {accuracy_percent:.2f}%")
                    elif accuracy_percent >= 60:
                        st.info(f"⚠️ Good accuracy: {accuracy_percent:.2f}%")
                    else:
                        st.warning(f"⚠️ Low accuracy: {accuracy_percent:.2f}%")
                    
                    # Per-class metrics
                    if "per_class" in metrics:
                        st.markdown("### Per-Class Metrics (Matched Samples)")
                        class_data = []
                        for label, class_metrics in metrics["per_class"].items():
                            if "precision" in class_metrics:
                                class_data.append({
                                    "Class": label,
                                    "Precision": f"{class_metrics['precision']*100:.2f}%",
                                    "Recall": f"{class_metrics['recall']*100:.2f}%",
                                    "F1-Score": f"{class_metrics['f1_score']*100:.2f}%",
                                    "Support": class_metrics['support']
                                })
                        if class_data:
                            class_df = pd.DataFrame(class_data)
                            st.dataframe(class_df, use_container_width=True, hide_index=True)
                    
                    # Confusion Matrix
                    if "confusion_matrix" in metrics:
                        st.markdown("### Confusion Matrix (Matched Samples)")
                        cm = np.array(metrics["confusion_matrix"])
                        cm_fig = plot_confusion_matrix(cm)
                        if cm_fig:
                            st.pyplot(cm_fig)
                        else:
                            cm_df = pd.DataFrame(
                                cm,
                                index=["Actual: Negative", "Actual: Neutral", "Actual: Positive"],
                                columns=["Pred: Negative", "Pred: Neutral", "Pred: Positive"]
                            )
                            st.dataframe(cm_df)
                    
                    # Detailed comparison for matched
                    st.markdown("### Matched Samples Comparison")
                    matched_comparison = []
                    for text, actual, pred in zip(matched_texts_list, matched_actuals, matched_predictions):
                        matched_comparison.append({
                            "Text": text[:100] + "..." if len(text) > 100 else text,
                            "Actual (Ref)": actual,
                            "Predicted (Test)": pred,
                            "Match": "✅" if actual == pred else "❌"
                        })
                    
                    if matched_comparison:
                        matched_df = pd.DataFrame(matched_comparison)
                        match_filter = st.selectbox("Filter matched samples:", ["All", "Correct", "Incorrect"], key="matched_filter")
                        if match_filter == "Correct":
                            matched_df = matched_df[matched_df["Match"] == "✅"]
                        elif match_filter == "Incorrect":
                            matched_df = matched_df[matched_df["Match"] == "❌"]
                        st.dataframe(matched_df.head(50), use_container_width=True, hide_index=True)
                
                # Show unmatched predictions
                if unmatched_predictions:
                    st.markdown("### Unmatched Samples (New Predictions)")
                    st.info(f"Found {len(unmatched_predictions)} texts in test dataset that don't match reference dataset. These are new predictions.")
                    
                    unmatched_df = pd.DataFrame({
                        "Text": [t[:100] + "..." if len(t) > 100 else t for t in unmatched_texts],
                        "Predicted Sentiment": unmatched_predictions
                    })
                    st.dataframe(unmatched_df.head(50), use_container_width=True, hide_index=True)
                
                # Download results
                st.markdown("---")
                st.markdown("### 💾 Download Results")
                
                download_data = []
                for text, pred in zip(test_texts, predicted_labels):
                    actual = ref_text_to_label.get(text.lower().strip(), "N/A (Not in reference)")
                    download_data.append({
                        "text": text,
                        "predicted_sentiment": pred,
                        "actual_sentiment": actual,
                        "match_status": "Matched" if text.lower().strip() in ref_text_to_label else "Unmatched"
                    })
                
                results_df = pd.DataFrame(download_data)
                csv_bytes = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download comparison results as CSV",
                    data=csv_bytes,
                    file_name="dataset_comparison_results.csv",
                    mime="text/csv",
                    key="download_comparison_results"
                )
    
    elif reference_file is None and test_file is None:
        st.info("👆 Please upload both datasets to begin comparison")
    else:
        st.warning("⚠️ Please upload both reference and test datasets")

# ----------- Mode 6: Manual Text Input -----------
elif mode == "Manual Text Input":
    # Ensure model UI is set up (will try to load transformer)
    ensure_model_ui()

    text = st.text_area("Enter text:")
    if st.button("Analyze Text"):
        if text.strip():
            cleaned = clean_text(text)
            
            # Get predictions and probabilities using helper function
            probas = predict_proba_sentiment([cleaned])[0]
            
            pred = probas.argmax()
            confidence = probas.max() * 100
            
            # Show result with confidence
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Sentiment", labels[pred])  # type: ignore
            with col2:
                st.metric("Confidence", f"{confidence:.1f}%")
            
            # Show probability breakdown
            st.markdown("**Probability Breakdown:**")
            prob_df = pd.DataFrame({
                "Sentiment": ["Negative", "Neutral", "Positive"],
                "Probability": [f"{probas[0]*100:.1f}%", f"{probas[1]*100:.1f}%", f"{probas[2]*100:.1f}%"]
            })
            st.dataframe(prob_df, use_container_width=True, hide_index=True)
            
            # Visual indicator
            if confidence > 80:
                st.success(f"✅ High confidence prediction")
            elif confidence > 60:
                st.info(f"⚠️ Moderate confidence prediction")
            else:
                st.warning(f"⚠️ Low confidence prediction - results may be unreliable")