import streamlit as st
import pandas as pd
import joblib
import requests
import os
import platform
import re
from textblob import TextBlob

# Cloud Platform Resource Safety Net
if platform.system() == "Linux" or os.environ.get("STREAMLIT_SHARING_MODE"):
    os.environ["DISABLE_HEAVY_AI"] = "1"

import io
from threading import Thread, Event
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
from collections import Counter
import numpy as np
import subprocess
import socket
import time

@st.cache_resource(show_spinner="Starting AI Backend Server (this takes ~15 seconds)...")
def start_backend():
    """Starts the FastAPI backend automatically if not running (e.g., on Streamlit Cloud)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1)
    is_open = False
    try:
        is_open = (s.connect_ex(('127.0.0.1', 8000)) == 0)
    except Exception:
        pass
    finally:
        s.close()

    if not is_open:
        import sys
        import subprocess
        proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8000"])
        # Wait up to 60 seconds for heavy transformer models to load
        start_time = time.time()
        while time.time() - start_time < 60:
            s2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s2.settimeout(1)
            try:
                if s2.connect_ex(('127.0.0.1', 8000)) == 0:
                    time.sleep(2)  # Give uvicorn a moment to fully bind
                    return proc
            except Exception:
                pass
            finally:
                s2.close()
            time.sleep(2)
        return proc
    return True

# Start backend if needed
start_backend()

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


# All ML model initialization has been moved to the FastAPI backend

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
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

labels = {0: "Negative", 1: "Neutral", 2: "Positive"}

# Label mapping function for complex sentiment labels (same as in train_model.py)
POSITIVE_KEYWORDS = ['positive', 'joy', 'excitement', 'contentment', 'happiness', 'love', 'grateful', 'amazing', 
                     'excellent', 'great', 'wonderful', 'fantastic', 'happy', 'pleased', 'satisfied', 'delighted', 
                     'thrilled', 'ecstatic', 'elated', 'jubilant', 'cheerful', 'optimistic', 'hopeful', 'proud', 
                     'triumph', 'heartwarming', 'celebrating', 'victory', 'success', 'achievement', 'gratitude',
                     'elation', 'playful', 'serenity', 'bliss', 'euphoria', 'content', 'fulfilled', 'blessed',
                     'appreciative', 'thankful', 'inspired', 'motivated', 'energetic', 'enthusiastic', 'passionate',
                     'awe', 'pride', 'enthusiasm', 'determination', 'surprise', 'inspiration', 'hope', 'empowerment',
                     'admiration', 'compassion', 'tenderness', 'arousal', 'fulfillment', 'reverence', 'thrill',
                     'enchantment', 'amusement', 'anticipation', 'kind', 'empathetic', 'free-spirited', 'confident',
                     'satisfaction', 'accomplishment', 'harmony', 'creativity', 'wonder', 'adventure', 'affection',
                     'adoration', 'zest', 'whimsy', 'radiance', 'rejuvenation', 'resilience', 'exploration',
                     'captivation', 'tranquility', 'mischievous', 'motivation', 'appreciation', 'confidence',
                     'wonderment', 'optimism', 'intrigue', 'mindfulness', 'elegance', 'melodic', 'innerjourney',
                     'freedom', 'dazzle', 'adrenaline', 'artisticburst', 'spark', 'marvel', 'positivity', 'kindness',
                     'friendship', 'amazement', 'romance', 'grandeur', 'energy', 'celebration', 'charm', 'ecstasy',
                     'colorful', 'connection', 'iconic', 'engagement', 'touched', 'solace', 'breakthrough',
                     'vibrancy', 'relief', 'sympathy']

NEGATIVE_KEYWORDS = ['negative', 'sad', 'angry', 'frustrated', 'disappointed', 'terrible', 'awful', 'bad', 'hate', 
                     'worst', 'horrible', 'disgusting', 'depressed', 'anxious', 'worried', 'fear', 'stress', 
                     'pressure', 'obstacle', 'problem', 'difficulty', 'challenge', 'failure', 'loss', 'pain', 
                     'suffering', 'grief', 'sorrow', 'despair', 'hopeless', 'bitterness', 'loneliness', 
                     'embarrassed', 'despair', 'hate', 'bitterness', 'resentment', 'rage', 'fury', 'annoyance',
                     'irritation', 'disgust', 'contempt', 'shame', 'guilt', 'regret', 'remorse', 'melancholy',
                     'gloom', 'misery', 'anguish', 'torment', 'agony', 'distress', 'trouble', 'hardship',
                     'anger', 'confusion', 'numbness', 'ambivalence', 'betrayal', 'boredom', 'overwhelmed',
                     'desolation', 'bitter', 'jealousy', 'jealous', 'devastated', 'envious', 'dismissive',
                     'heartbreak', 'anxiety', 'intimidation', 'helplessness', 'envy', 'yearning', 'apprehensive',
                     'isolation', 'disappointment', 'emotionalstorm', 'exhaustion', 'darkness', 'desperation',
                     'ruins', 'heartache', 'solitude', 'miscalculation']

NEUTRAL_KEYWORDS = ['neutral', 'okay', 'fine', 'average', 'normal', 'regular', 'standard', 'typical', 'ordinary', 
                    'moderate', 'balanced', 'calm', 'indifferent', 'unbiased', 'objective', 'factual', 'informative',
                    'curiosity', 'wondering', 'questioning', 'contemplative', 'reflective', 'thoughtful', 'pensive',
                    'contemplative', 'analytical', 'logical', 'rational', 'practical', 'matter-of-fact',
                    'acceptance', 'indifference', 'reflection', 'contemplation', 'emotion', 'journey', 'immersion',
                    'nostalgia']

def map_sentiment_label(label):
    """
    Map complex sentiment labels to 3 classes: Negative, Neutral, Positive.
    Uses keyword matching to categorize labels (same function as in train_model.py).
    Returns string labels: "Negative", "Neutral", or "Positive"
    """
    label_str = str(label).strip().lower()
    
    # Direct mapping if already in standard format
    label_map = {"negative": "Negative", "neutral": "Neutral", "positive": "Positive"}
    if label_str in label_map:
        return label_map[label_str]
    
    # Check for positive keywords
    for keyword in POSITIVE_KEYWORDS:
        if keyword in label_str:
            return "Positive"
    
    # Check for negative keywords
    for keyword in NEGATIVE_KEYWORDS:
        if keyword in label_str:
            return "Negative"
    
    # Check for neutral keywords
    for keyword in NEUTRAL_KEYWORDS:
        if keyword in label_str:
            return "Neutral"
    
    # Try to parse as integer
    try:
        iv = int(label_str)
        if iv in [0, 1, 2]:
            return {0: "Negative", 1: "Neutral", 2: "Positive"}[iv]
        if iv == 4:
            return "Positive"
    except (ValueError, TypeError):
        pass
    
    # TextBlob fallback
    try:
        from textblob import TextBlob
        tb = TextBlob(label_str)
        if tb.sentiment.polarity > 0.05:
            return "Positive"
        elif tb.sentiment.polarity < -0.05:
            return "Negative"
    except:
        pass
    
    # Default to neutral if unclear
    return "Neutral"

def predict_sentiment(texts):
    """Predict sentiment using custom model if exists, else backend API."""
    if isinstance(texts, str):
        texts = [texts]
        
    try:
        import joblib
        import os
        model_path = os.path.join(os.path.dirname(__file__), "model.joblib")
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            return model.predict(texts).tolist()
    except Exception as e:
        print(f"Failed to use custom model: {e}")
            
    try:
        import requests
        res = requests.post("http://127.0.0.1:8000/predict", json={"texts": texts}, timeout=60)
        if res.status_code == 200:
            preds = res.json().get("predictions", [])
            text_to_num = {"Negative": 0, "Neutral": 1, "Positive": 2}
            results = [text_to_num.get(p, 1) for p in preds]
            return results
    except Exception as e:
        print(f"Backend API prediction failed: {e}")
    
    return [1] * len(texts)

def predict_proba_sentiment(texts):
    """Predict probabilities using custom model if exists, else backend API."""
    if isinstance(texts, str):
        texts = [texts]
        
    try:
        import joblib
        import os
        model_path = os.path.join(os.path.dirname(__file__), "model.joblib")
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            return model.predict_proba(texts)
    except Exception as e:
        print(f"Failed to use custom model for proba: {e}")
            
    try:
        import requests
        import numpy as np
        res = requests.post("http://127.0.0.1:8000/predict", json={"texts": texts}, timeout=60)
        if res.status_code == 200:
            return np.array(res.json().get("probabilities", []))
    except Exception as e:
        print(f"Backend API probability prediction failed: {e}")
    
    import numpy as np
    return np.array([[0.2, 0.6, 0.2]] * len(texts))

def ensure_model_ui():
    """In-UI helper: check if backend API is reachable."""
    try:
        res = requests.get("http://127.0.0.1:8000/", timeout=5)
        if res.status_code == 200:
            return True
    except requests.exceptions.RequestException:
        st.error("❌ Backend API is not reachable. Please ensure the FastAPI server is running on port 8000.")
        st.info("Run: `uvicorn backend.main:app --reload` in your terminal.")
        return False
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
    """Read a CSV or Excel file and try to detect if the real header is on a later row."""
    try:
        is_excel = uploaded.name.lower().endswith(('.xlsx', '.xls'))
        
        # Read preview
        if is_excel:
            if hasattr(uploaded, 'seek'): uploaded.seek(0)
            preview = pd.read_excel(uploaded, header=None, nrows=10, dtype=str)
            if hasattr(uploaded, 'seek'): uploaded.seek(0)
        else:
            if hasattr(uploaded, 'read'):
                uploaded.seek(0)
                sample_text = uploaded.read().decode('utf-8', errors='ignore')
                import io
                preview_buf = io.StringIO(sample_text)
            else:
                preview_buf = uploaded
            preview = pd.read_csv(preview_buf, header=None, nrows=10, dtype=str, keep_default_na=False)

        header_row = None
        for i, row in preview.iterrows():
            row_vals = ' '.join([str(x).lower() for x in row.tolist()])
            if re.search(r'text', row_vals):
                header_row = i
                break

        if header_row is None:
            first0 = str(preview.iloc[0, 0]).lower()
            if 'social media' in first0 or 'sentiments' in first0 or 'social' in first0:
                header_row = 1

        # Helper to read full file safely
        def safe_read_full_file(file_obj, header_idx):
            kwargs = {'header': header_idx, 'dtype': str}
            if not is_excel:
                kwargs['keep_default_na'] = False
                
            if hasattr(file_obj, 'size') and file_obj.size > 50 * 1024 * 1024:
                kwargs['nrows'] = 50000
                st.warning("File is extremely large (>50MB). Only loading the first 50,000 rows to prevent Streamlit memory crashes.")
                
            if is_excel:
                if hasattr(file_obj, 'seek'): file_obj.seek(0)
                return pd.read_excel(file_obj, **kwargs)
                
            encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
            for enc in encodings:
                try:
                    if hasattr(file_obj, 'seek'): file_obj.seek(0)
                    return pd.read_csv(file_obj, encoding=enc, **kwargs)
                except UnicodeDecodeError:
                    continue
            if hasattr(file_obj, 'seek'): file_obj.seek(0)
            return pd.read_csv(file_obj, encoding='utf-8', encoding_errors='replace', **kwargs)

        df = safe_read_full_file(uploaded, header_row if header_row is not None else 0)

        # Excel can return NaNs for empty strings, so fill them for text processing
        if is_excel:
            df = df.fillna('')

        return df, header_row
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None, None

st.set_page_config(page_title="Social Media Sentiment Analyzer", layout="centered", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');
    html, body, .stApp, .stMarkdown, p, h1, h2, h3, h4, h5, h6, label, input, button { font-family: 'Roboto', sans-serif !important; }
    /* Protect Material Icons and internal icon classes from being overridden by our font */
    i, .material-icons, [class*="icon"], [class*="stIcon"] { font-family: 'Material Icons', 'Material Icons Round', inherit !important; }
    body, .stApp { background-color: #f8f9fa !important; color: #334155; }
    .main { 
        background-color: #ffffff !important; 
        border: 1px solid #e2e8f0; 
        border-radius: 8px; padding: 2rem; 
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06); 
    }
    h1, h2, h3, h4 { color: #0f172a !important; font-weight: 700; }
    h1 { color: #1e3a8a !important; text-align: center; border-bottom: 2px solid #e2e8f0; padding-bottom: 10px; margin-bottom: 20px; }
    .stButton>button { 
        background-color: #2563eb !important; 
        color: #ffffff !important; font-weight: 500; border-radius: 4px; border: 1px solid #1d4ed8; 
        padding: 0.5rem 1.5rem; transition: background-color 0.2s ease; 
    }
    .stButton>button:hover { background-color: #1d4ed8 !important; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); }
    .stFileUploader>div>div { background-color: #f8fafc !important; border: 1px dashed #cbd5e1 !important; border-radius: 6px; }
    .stTextInput>div>div>input, .stTextArea>div>textarea { background-color: #ffffff !important; color: #334155 !important; border: 1px solid #cbd5e1; border-radius: 4px; }
    .stTextInput>div>div>input:focus, .stTextArea>div>textarea:focus { border-color: #2563eb; box-shadow: 0 0 0 1px #2563eb; }
    .stMetric { background-color: #ffffff; padding: 1rem; border-radius: 6px; border: 1px solid #e2e8f0; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }
    .stMetric label { color: #64748b !important; font-size: 0.875rem !important; text-transform: uppercase; letter-spacing: 0.05em; }
    .stMetric [data-testid="stMetricValue"] { color: #0f172a !important; font-weight: 700 !important; font-size: 1.8rem !important; }
    .stDataFrame { border: 1px solid #e2e8f0; border-radius: 4px; }
    div[data-testid="stSidebar"] { background-color: #f1f5f9 !important; border-right: 1px solid #e2e8f0; }
    .stAlert { border-radius: 4px; border-left: 4px solid #2563eb; background-color: #eff6ff; color: #1e3a8a; }
    </style>
    """,
    unsafe_allow_html=True
)



st.sidebar.header("Choose Mode")
mode = st.sidebar.selectbox(
    "Select Analysis Mode",
    [
        "Analyze Dataset",
        "Analyze Image/Screenshot",
        "Manual Text Input",
        "Prediction History (Database)",
        "Train Custom Model"
    ],
    key="mode_selectbox"
)

st.sidebar.markdown("---")

st.sidebar.markdown("---")
st.sidebar.header("📊 Backend Server Status")
try:
    # Quick check if backend is up
    _res = requests.get("http://127.0.0.1:8000/", timeout=2)
    if _res.status_code == 200:
        st.sidebar.success("✅ FastAPI Backend Active")
        st.sidebar.caption("Status: Online<br>Database connected", unsafe_allow_html=True)
    else:
        st.sidebar.warning("⚠️ Backend returned unexpected status")
except:
    st.sidebar.error("❌ Backend Offline")
    st.sidebar.caption("Please start the FastAPI server on port 8000")

# Add OCR Info section
st.sidebar.markdown("---")
st.sidebar.header("📸 OCR Status")
if OCR_AVAILABLE:
    st.sidebar.success("✅ OCR Available")
    st.sidebar.caption("Image text extraction<br>enabled with PyTesseract", unsafe_allow_html=True)
else:
    st.sidebar.info("ℹ️ OCR Not Available")
    st.sidebar.caption("Install: pip install pytesseract Pillow")

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
    
    # Remove special characters but keep spaces, letters, numbers, and basic punctuation
    text = re.sub(r'[^A-Za-z0-9\s.,!?]', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text



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
    """Initialize OCR reader (compatibility placeholder)."""
    return OCR_AVAILABLE

def extract_text_from_image(image_file):
    """
    Extract text from an image using PyTesseract.
    
    Args:
        image_file: Uploaded image file (Streamlit UploadedFile)
    
    Returns:
        List of extracted text strings
    """
    if not OCR_AVAILABLE:
        return []
    
    try:
        # Read image
        image = Image.open(image_file)
        
        # Perform OCR
        text = pytesseract.image_to_string(image)
        
        # Extract text from results
        extracted_texts = []
        if text and text.strip():
            extracted_texts.append(text.strip())
        
        return extracted_texts
    except Exception as e:
        if "tesseract is not installed" in str(e).lower() or "not correctly configured" in str(e).lower():
            st.error("Tesseract Engine is not installed on the system. OCR cannot proceed.")
        else:
            st.error(f"Error extracting text from image: {e}")
        return []

# ----------- Mode 1: Dataset Analyzer -----------
if mode == "Analyze Dataset":
    st.subheader("📊 Dataset Sentiment Analysis & Auto-Trainer")
    st.markdown("Upload a CSV file. If it contains **labels**, we will automatically train a high-accuracy custom model on it and display the accuracy meter. If it has **no labels**, we will analyze the text and give you the predictions.")
    
    uploaded_file = st.file_uploader("Upload your dataset (CSV/Excel)", type=["csv", "xlsx", "xls"], key="dataset_uploader")
    if uploaded_file is not None:
        df, header_row = read_csv_with_header_detection(uploaded_file)
        
        if df is None:
            st.error("Could not parse uploaded CSV. Please check the file format.")
        else:
            st.success(f"✅ Dataset loaded successfully: {len(df)} rows")
            
            # Detect text and label columns
            cols = list(df.columns)
            text_col = detect_text_column(df, exclude_numerical=True)
            
            sentiment_col = None
            for c in cols:
                if 'sentiment' in str(c).lower() or 'label' in str(c).lower():
                    sentiment_col = c
                    break
            
            st.markdown("### Step 2: Select Columns")
            col1, col2 = st.columns(2)
            with col1:
                t_idx = cols.index(text_col) if text_col in cols else 0
                selected_text = st.selectbox("Text Column", options=cols, index=t_idx, key="data_text")
            with col2:
                # Add "None (Predict Only)" option for labels
                label_options = ["None (Predict Only)"] + cols
                s_idx = label_options.index(sentiment_col) if sentiment_col in label_options else 0
                selected_label = st.selectbox("Label Column (Select 'None' to just predict)", options=label_options, index=s_idx, key="data_label")
                
            st.markdown("### Preview")
            if selected_label != "None (Predict Only)":
                st.dataframe(df[[selected_text, selected_label]].head(5))
            else:
                st.dataframe(df[[selected_text]].head(5))
                
            if st.button("Process Dataset", key="process_dataset_btn"):
                texts = df[selected_text].astype(str).tolist()
                
                if selected_label != "None (Predict Only)":
                    # --- AUTO-TRAIN AND ACCURACY METER MODE ---
                                        # --- EVALUATION AND ACCURACY METER MODE ---
                    with st.spinner("Evaluating dataset with RoBERTa model..."):
                        try:
                            labels_list = df[selected_label].apply(map_sentiment_label).tolist()
                            valid_data = [(t, l) for t, l in zip(texts, labels_list) if l in ["Negative", "Neutral", "Positive"]]
                            
                            if not valid_data:
                                st.error("No valid labels found in the selected column.")
                                st.stop()
                                
                            if len(valid_data) > 15000:
                                import random
                                valid_data = random.sample(valid_data, 15000)
                                st.warning("Dataset is extremely large! Randomly sampled 15,000 rows for analysis to prevent server memory crashes.")
                                
                            X = [d[0] for d in valid_data]
                            y = [d[1] for d in valid_data]
                            y_num = [{"Negative": 0, "Neutral": 1, "Positive": 2}[lbl] for lbl in y]
                            
                            import hashlib
                            import joblib
                            import os
                            
                            # Cache predictions so we don't re-run RoBERTa on the same dataset
                            dataset_hash = hashlib.md5("".join(X).encode('utf-8')).hexdigest()
                            cache_path = os.path.join(os.path.dirname(__file__), "eval_cache_v2.joblib")
                            
                            y_pred_num = None
                            if os.path.exists(cache_path):
                                try:
                                    cache = joblib.load(cache_path)
                                    if dataset_hash in cache:
                                        y_pred_num = cache[dataset_hash]
                                except Exception:
                                    pass
                            
                            if y_pred_num is None:
                                y_pred_num = []
                                chunk_size = 50
                                progress_bar = st.progress(0)
                                for i in range(0, len(X), chunk_size):
                                    chunk = X[i:i+chunk_size]
                                    y_pred_num.extend(predict_sentiment(chunk))
                                    progress_bar.progress(min(1.0, (i + len(chunk)) / len(X)))
                                progress_bar.empty()
                                
                                try:
                                    cache = joblib.load(cache_path) if os.path.exists(cache_path) else {}
                                    cache[dataset_hash] = y_pred_num
                                    joblib.dump(cache, cache_path)
                                except Exception:
                                    pass
                                
                            from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
                            import matplotlib.pyplot as plt
                            import seaborn as sns
                            
                            acc = accuracy_score(y_num, y_pred_num)
                            try:
                                f1 = f1_score(y_num, y_pred_num, average='weighted')
                            except Exception:
                                f1 = 0.0
                            
                            st.markdown("---")
                            st.markdown("## 📊 Dataset Overview & Accuracy Results")
                            
                            # Metrics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("RoBERTa Accuracy", f"{acc*100:.2f}%")
                            with col2:
                                st.metric("F1 Score (Weighted)", f"{f1*100:.2f}%")
                            with col3:
                                st.metric("Total Valid Samples", len(valid_data))
                                
                            # Prepare results dataframe
                            pred_labels_text = [{0: "Negative", 1: "Neutral", 2: "Positive"}.get(p, "Neutral") for p in y_pred_num]
                            results_df = pd.DataFrame({
                                "Text": X,
                                "Actual_Sentiment": y,
                                "Predicted_Sentiment": pred_labels_text
                            })
                            
                            st.markdown("---")
                            st.markdown("### 📊 Sentiment Distribution Graph")
                            sentiment_counts = results_df["Predicted_Sentiment"].value_counts()
                            st.bar_chart(sentiment_counts, color="#3b82f6")
                            # Confusion Matrix and Graphs
                            st.markdown("### Model Performance Visualizations")
                            col_cm, col_chart = st.columns(2)
                            
                            with col_cm:
                                st.markdown("**Confusion Matrix**")
                                cm = confusion_matrix(y_num, y_pred_num)
                                fig, ax = plt.subplots(figsize=(5, 4))
                                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                                            xticklabels=["Negative", "Neutral", "Positive"], 
                                            yticklabels=["Negative", "Neutral", "Positive"])
                                plt.ylabel('Actual')
                                plt.xlabel('Predicted')
                                st.pyplot(fig)
                                
                            with col_chart:
                                st.markdown("**Prediction Distribution**")
                                counts = results_df["Predicted_Sentiment"].value_counts()
                                render_pie_chart(counts, title="")
                                
                            # Classification Report
                            st.markdown("### Detailed Metrics")
                            try:
                                report = classification_report(y_num, y_pred_num, target_names=["Negative", "Neutral", "Positive"])
                                st.code(report)
                            except Exception:
                                st.write("Could not generate classification report.")
                            
                            st.markdown("### Prediction Previews")
                            st.dataframe(results_df.head(20))
                                
                            csv_bytes = results_df.to_csv(index=False).encode("utf-8")
                            st.download_button("Download Full Results (CSV)", data=csv_bytes, file_name="accuracy_results.csv", mime="text/csv")
                        except Exception as e:
                            st.error(f"Evaluation failed: {e}")
                else:
                    # --- PREDICTION ONLY MODE ---
                    with st.spinner("Analyzing sentiments..."):
                        if not ensure_model_ui():
                            st.stop()
                            
                        chunk_size = max(1, len(texts) // 10)
                        preds = []
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for i in range(0, len(texts), chunk_size):
                            chunk = texts[i:i+chunk_size]
                            chunk_preds = predict_sentiment(chunk)
                            preds.extend(chunk_preds)
                            progress = min(1.0, (i + len(chunk)) / len(texts))
                            progress_bar.progress(progress)
                            status_text.text(f"Analyzed {len(preds)}/{len(texts)} texts...")
                            
                        progress_bar.empty()
                        status_text.empty()
                        
                        df["Predicted Sentiment"] = [labels.get(p, "Neutral") for p in preds]
                        
                        st.markdown("### Analysis Results")
                        st.dataframe(df.head(20))
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            counts = df["Predicted Sentiment"].value_counts()
                            render_pie_chart(counts, title="Sentiment Distribution")
                        with col2:
                            st.markdown("**Keyword Cloud**")
                            generate_wordcloud(texts, "Common Words in Dataset")
                            
                        csv_bytes = df.to_csv(index=False).encode("utf-8")
                        st.download_button("Download Predictions (CSV)", data=csv_bytes, file_name="predictions.csv", mime="text/csv")

elif mode == "Analyze Image/Screenshot":
    st.subheader("📸 Image & Screenshot Sentiment Analysis")
    
    # Check OCR availability
    if not OCR_AVAILABLE:
        st.error("""
        **OCR (Optical Character Recognition) is not available.**
        
        **To enable OCR**:
        1. Install required packages: `pip install pytesseract Pillow`
        2. Ensure Tesseract Engine is installed on your system (Cloud: packages.txt handles this)
        3. Restart the app
        
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
                        if ensure_model_ui():
                            # Call the new FastAPI backend seamlessly
                            pred_int = predict_sentiment([cleaned_text])[0]
                            sentiment_label = labels[pred_int]
                            probas = predict_proba_sentiment([cleaned_text])[0]
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
                                        seg_pred_int = predict_sentiment([cleaned_segment])[0]
                                        seg_label = labels[seg_pred_int]
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
                            st.warning("Backend API unreachable.")
                
                # Download results (only if analysis was performed)
                if full_text.strip():
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

# ----------- Mode 6: Prediction History (Database) -----------
elif mode == "Prediction History (Database)":
    st.subheader("🗄️ Prediction History (Database)")
    st.markdown("View the history of all texts analyzed by the system, saved automatically in our local SQLite database.")
    
    try:
        response = requests.get("http://127.0.0.1:8000/history?limit=100", timeout=10)
        if response.status_code == 200:
            history = response.json()
            if not history:
                st.info("No predictions found in the database yet. Try analyzing some text!")
            else:
                df_history = pd.DataFrame(history)
                # If emotion exists in the database response, add it to the view
                columns_to_show = ["id", "timestamp", "sentiment"]
                if "emotion" in df_history.columns:
                    columns_to_show.append("emotion")
                columns_to_show.append("text")
                st.dataframe(df_history[columns_to_show], use_container_width=True, hide_index=True)
                
                # Show quick pie chart of historical counts
                st.markdown("### 📊 Historical Analytics")
                import plotly.express as px
                
                col1, col2 = st.columns(2)
                with col1:
                    counts = df_history["sentiment"].value_counts().reset_index()
                    counts.columns = ["sentiment", "count"]
                    fig = px.pie(counts, values='count', names='sentiment', title='Sentiment Distribution', color='sentiment', color_discrete_map={'Positive':'#2ecc71','Neutral':'#f1c40f','Negative':'#e74c3c'})
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    if "emotion" in df_history.columns:
                        em_counts = df_history["emotion"].value_counts().reset_index()
                        em_counts.columns = ["emotion", "count"]
                        fig_em = px.pie(em_counts, values='count', names='emotion', hole=0.3, title='Emotion Distribution', color_discrete_sequence=px.colors.qualitative.Pastel)
                        st.plotly_chart(fig_em, use_container_width=True)
        else:
            st.error(f"Failed to fetch history: {response.text}")
    except Exception as e:
        st.error(f"Database unreachable. Please ensure FastAPI server is running. Error: {e}")
# ----------- Mode 7: Train Custom Model -----------
elif mode == "Train Custom Model":
    st.subheader("🛠️ Train Custom Model (Fine-tune RoBERTa)")
    st.markdown("Upload a CSV dataset with `text` and `label` columns to fine-tune a RoBERTa model.")
    
    uploaded_file = st.file_uploader("Upload training dataset (CSV)", type=["csv"], key="train_uploader")
    if uploaded_file is not None:
        df, header_row = read_csv_with_header_detection(uploaded_file)
        if df is not None:
            st.success(f"✅ Dataset loaded successfully: {len(df)} rows")
            cols = list(df.columns)
            text_col = detect_text_column(df, exclude_numerical=True)
            
            col1, col2 = st.columns(2)
            with col1:
                t_idx = cols.index(text_col) if text_col in cols else 0
                selected_text = st.selectbox("Text Column", options=cols, index=t_idx, key="train_text")
            with col2:
                label_options = cols
                s_idx = 1 if len(cols) > 1 else 0
                selected_label = st.selectbox("Label Column", options=label_options, index=s_idx, key="train_label")
                
            if st.button("Start Fine-Tuning"):
                with st.spinner("Preparing data and model (this will take time)..."):
                    try:
                        texts = df[selected_text].astype(str).tolist()
                        
                        labels_list = [map_sentiment_label(lbl) for lbl in df[selected_label]]
                        valid_data = [(t, l) for t, l in zip(texts, labels_list) if l in ["Negative", "Neutral", "Positive"]]
                        
                        if not valid_data:
                            st.error("No valid labels found. Labels must map to Negative, Neutral, or Positive.")
                        else:
                            X = [d[0] for d in valid_data]
                            y = [d[1] for d in valid_data]
                            label_mapping = {"Negative": 0, "Neutral": 1, "Positive": 2}
                            y_num = [label_mapping[lbl] for lbl in y]
                            
                            st.info("Splitting dataset into 70% training and 30% testing...")
                            from sklearn.model_selection import train_test_split
                            train_texts, test_texts, train_labels, test_labels = train_test_split(
                                X, y_num, test_size=0.3, random_state=42
                            )
                            
                            st.info(f"Train size: {len(train_texts)} | Test size: {len(test_texts)}")
                            
                            st.info("Loading RoBERTa tokenizer...")
                            from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
                            from torch.utils.data import Dataset
                            import torch
                            
                            tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
                            
                            class CustomDataset(Dataset):
                                def __init__(self, encodings, labels):
                                    self.encodings = encodings
                                    self.labels = labels
                                def __getitem__(self, idx):
                                    item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                                    item['labels'] = torch.tensor(self.labels[idx])
                                    return item
                                def __len__(self):
                                    return len(self.labels)

                            train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
                            test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)
                            
                            train_dataset = CustomDataset(train_encodings, train_labels)
                            test_dataset = CustomDataset(test_encodings, test_labels)
                            
                            st.info("Loading RoBERTa model...")
                            model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3)
                            
                            training_args = TrainingArguments(
                                output_dir='./results',
                                num_train_epochs=3,
                                per_device_train_batch_size=8,
                                per_device_eval_batch_size=8,
                                evaluation_strategy="epoch",
                                save_strategy="epoch",
                                logging_steps=10,
                                load_best_model_at_end=True,
                            )
                            
                            from sklearn.metrics import accuracy_score
                            import numpy as np
                            def compute_metrics(eval_pred):
                                logits, labels = eval_pred
                                predictions = np.argmax(logits, axis=-1)
                                return {"accuracy": accuracy_score(labels, predictions)}
                            
                            trainer = Trainer(
                                model=model,
                                args=training_args,
                                train_dataset=train_dataset,
                                eval_dataset=test_dataset,
                                compute_metrics=compute_metrics
                            )
                            
                            st.info("Starting training...")
                            trainer.train()
                            
                            st.success("Training completed!")
                            
                            st.info("Evaluating on 30% test set...")
                            eval_results = trainer.evaluate()
                            st.write(eval_results)
                            
                            trainer.save_model("./custom_roberta_model")
                            tokenizer.save_pretrained("./custom_roberta_model")
                            st.success("Model saved to ./custom_roberta_model directory!")
                            
                    except Exception as e:
                        st.error(f"Error during training: {e}")
