import streamlit as st
import pandas as pd
import joblib
import requests
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
    import easyocr
    from PIL import Image
    import io
    OCR_AVAILABLE = True
    ocr_reader = None
except ImportError:
    OCR_AVAILABLE = False
    ocr_reader = None

labels = {0: "Negative", 1: "Neutral", 2: "Positive"}

# Label mapping function for complex sentiment labels (same as in train_model.py)
POSITIVE_KEYWORDS = ['positive', 'joy', 'excitement', 'contentment', 'happiness', 'love', 'grateful', 'amazing', 
                     'excellent', 'great', 'wonderful', 'fantastic', 'happy', 'pleased', 'satisfied', 'delighted', 
                     'thrilled', 'ecstatic', 'elated', 'jubilant', 'cheerful', 'optimistic', 'hopeful', 'proud', 
                     'triumph', 'heartwarming', 'celebrating', 'victory', 'success', 'achievement', 'gratitude',
                     'elation', 'playful', 'serenity', 'bliss', 'euphoria', 'content', 'fulfilled', 'blessed',
                     'appreciative', 'thankful', 'inspired', 'motivated', 'energetic', 'enthusiastic', 'passionate']

NEGATIVE_KEYWORDS = ['negative', 'sad', 'angry', 'frustrated', 'disappointed', 'terrible', 'awful', 'bad', 'hate', 
                     'worst', 'horrible', 'disgusting', 'depressed', 'anxious', 'worried', 'fear', 'stress', 
                     'pressure', 'obstacle', 'problem', 'difficulty', 'challenge', 'failure', 'loss', 'pain', 
                     'suffering', 'grief', 'sorrow', 'despair', 'hopeless', 'bitterness', 'loneliness', 
                     'embarrassed', 'despair', 'hate', 'bitterness', 'resentment', 'rage', 'fury', 'annoyance',
                     'irritation', 'disgust', 'contempt', 'shame', 'guilt', 'regret', 'remorse', 'melancholy',
                     'gloom', 'misery', 'anguish', 'torment', 'agony', 'distress', 'trouble', 'hardship']

NEUTRAL_KEYWORDS = ['neutral', 'okay', 'fine', 'average', 'normal', 'regular', 'standard', 'typical', 'ordinary', 
                    'moderate', 'balanced', 'calm', 'indifferent', 'unbiased', 'objective', 'factual', 'informative',
                    'curiosity', 'wondering', 'questioning', 'contemplative', 'reflective', 'thoughtful', 'pensive',
                    'contemplative', 'analytical', 'logical', 'rational', 'practical', 'matter-of-fact']

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
    
    # Try to parse as integer (0, 1, or 2)
    try:
        iv = int(label_str)
        if iv in [0, 1, 2]:
            return labels[iv]
    except (ValueError, TypeError):
        pass
    
    # Default to neutral if unclear
    return "Neutral"

def predict_sentiment(texts):
    """Predict sentiment via the FastAPI backend API with batching."""
    if isinstance(texts, str):
        texts = [texts]
    
    batch_size = 50
    results = []
    
    try:
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            response = requests.post("http://127.0.0.1:8000/predict", json={"texts": batch}, timeout=60)
            if response.status_code == 200:
                predictions = response.json().get("predictions", [])
                # Map text labels back to integers for UI compatibility
                text_to_num = {"Negative": 0, "Neutral": 1, "Positive": 2}
                results.extend([text_to_num.get(p, 1) for p in predictions])
            else:
                results.extend([1] * len(batch))
        return results
    except Exception as e:
        if len(results) == 0:
            st.error(f"Backend API Error: {e}")
            
    # Absolute fallback if API fails for remaining
    remaining = len(texts) - len(results)
    if remaining > 0:
        results.extend([1] * remaining)
    return results

def predict_proba_sentiment(texts):
    """Predict sentiment probabilities via the FastAPI backend with batching."""
    if isinstance(texts, str):
        texts = [texts]
    
    batch_size = 50
    results = []
    
    try:
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            response = requests.post("http://127.0.0.1:8000/predict", json={"texts": batch}, timeout=60)
            if response.status_code == 200:
                probs = response.json().get("probabilities", [])
                if probs:
                    results.extend(probs)
                else:
                    results.extend([[0.2, 0.6, 0.2]] * len(batch))
            else:
                results.extend([[0.2, 0.6, 0.2]] * len(batch))
        
        if results:
            return np.array(results)
    except:
        pass
    
    # Fallback to neutral probabilities if API fails for remaining
    remaining = len(texts) - len(results)
    if remaining > 0:
        results.extend([[0.2, 0.6, 0.2]] * remaining)
    
    return np.array(results)

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
        "Analyze Social Media Link",
        "Analyze Image/Screenshot",
        "Accuracy Meter/Validation",
        "Manual Text Input",
        "Prediction History (Database)"
    ],
    key="mode_selectbox"
)

# Add Model Info section
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
    
    # Remove special characters but keep spaces, letters, numbers, and basic punctuation
    text = re.sub(r'[^A-Za-z0-9\s.,!?]', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

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
                # Get original texts before cleaning for validation
                original_texts = df[text_col].astype(str).tolist()
                texts = []
                valid_indices = []
                filtered_count = 0
                
                for i, original_text in enumerate(original_texts):
                    # Check original text first before cleaning
                    if original_text is None or pd.isna(original_text):
                        filtered_count += 1
                        continue
                    
                    original_str = str(original_text).strip()
                    if not original_str or original_str.lower() in ['nan', 'none', 'null', '']:
                        filtered_count += 1
                        continue
                    
                    # Only filter if it's clearly numeric-only (very strict check)
                    if is_numeric_only(original_str):
                        filtered_count += 1
                        continue
                    
                    # Clean the text for processing
                    cleaned = clean_text(original_str)
                    if cleaned:  # Only add if cleaning produced valid text
                        texts.append(cleaned)
                        valid_indices.append(i)
                    elif len(original_str) > 3:  # If original has content, use it even if cleaning removed some
                        texts.append(original_str)
                        valid_indices.append(i)
                    else:
                        filtered_count += 1
                
                if filtered_count > 0:
                    st.info(f"ℹ️ Filtered out {filtered_count} numeric-only or empty entries (not suitable for sentiment analysis)")
                
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
                
                # Ensure predictions were generated
                if not preds:
                    st.error("Failed to generate predictions. Please check your model.")
                    st.stop()
                
                df_filtered["Predicted"] = preds
                df_filtered["Predicted_Label"] = df_filtered["Predicted"].map(labels)  # type: ignore
                result_df = df_filtered
                st.success(f"Full dataset analysis complete - {len(texts)} valid entries analyzed")
            else:
                # Ensure model UI is set up (will try to load transformer)
                ensure_model_ui()
                
                # Use first N rows (deterministic) instead of random sampling
                head_df = df.head(int(sample_size)).copy()
                original_head_texts = head_df[text_col].astype(str).tolist()
                
                # Filter out numeric-only and empty texts (less aggressive)
                head_texts = []
                valid_indices = []
                filtered_count = 0
                
                for i, original_text in enumerate(original_head_texts):
                    # Check original text first before cleaning
                    if original_text is None or pd.isna(original_text):
                        filtered_count += 1
                        continue
                    
                    original_str = str(original_text).strip()
                    if not original_str or original_str.lower() in ['nan', 'none', 'null', '']:
                        filtered_count += 1
                        continue
                    
                    # Only filter if it's clearly numeric-only (very strict check)
                    if is_numeric_only(original_str):
                        filtered_count += 1
                        continue
                    
                    # Clean the text for processing
                    cleaned = clean_text(original_str)
                    if cleaned:  # Only add if cleaning produced valid text
                        head_texts.append(cleaned)
                        valid_indices.append(i)
                    elif len(original_str) > 3:  # If original has content, use it even if cleaning removed some
                        head_texts.append(original_str)
                        valid_indices.append(i)
                    else:
                        filtered_count += 1
                
                if filtered_count > 0:
                    st.info(f"ℹ️ Filtered out {filtered_count} numeric-only or empty entries (not suitable for sentiment analysis)")
                
                head_df_filtered = head_df.iloc[valid_indices].copy() if valid_indices else head_df.copy()
                
                if len(head_texts) == 0:
                    st.warning("⚠️ No valid text entries found after filtering. Please check your dataset.")
                    st.stop()
                
                # Ensure model is ready (delegated to backend)
                if not ensure_model_ui():
                    st.error("Backend API is unreachable.")
                    st.stop()
                
                head_preds = predict_sentiment(head_texts)
                if not head_preds:
                    st.error("Failed to generate predictions. Please check your model.")
                    st.stop()
                
                head_df_filtered["Predicted"] = head_preds
                head_df_filtered["Predicted_Label"] = head_df_filtered["Predicted"].map(labels)  # type: ignore
                result_df = head_df_filtered
                st.success(f"Quick scan (first {sample_size} rows) complete - {len(head_texts)} valid entries analyzed")

            # Only show results if analysis was performed
            if 'result_df' in locals() and result_df is not None and not result_df.empty:
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

# ----------- Mode 4: Accuracy Meter/Validation -----------
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
            
            # Validate labels (silently - don't show warning for non-standard labels)
            # The app will work with any labels, but only Positive/Negative/Neutral are used for accuracy calculation
            
            if st.button("Calculate Accuracy", key="calculate_accuracy_btn"):
                # Clean and prepare data
                texts = df_ref[text_col].astype(str).apply(clean_text).tolist()
                actual_labels = df_ref[sentiment_col].astype(str).str.strip().tolist()
                
                # Normalize labels using the same mapping function as training
                actual_labels = [map_sentiment_label(l) for l in actual_labels]
                
                # Predict sentiments
                with st.spinner("Predicting sentiments..."):
                    if not ensure_model_ui():
                        st.stop()
                    predicted_numeric = predict_sentiment(texts)
                    predicted_labels = [labels[p] for p in predicted_numeric]
                
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
                if "classification_report" in metrics:
                    st.markdown("### 📋 Classification Report")
                    if isinstance(metrics["classification_report"], str):
                        # Display as code block for better formatting
                        st.code(metrics["classification_report"], language=None)
                    else:
                        # If it's not a string, try to generate it
                        try:
                            if SKLEARN_METRICS_AVAILABLE:
                                # Convert labels to numeric for report generation
                                label_map = {"Negative": 0, "Neutral": 1, "Positive": 2}
                                y_true_num = [label_map.get(l, 1) if isinstance(l, str) else int(l) for l in actual_labels]
                                y_pred_num = [label_map.get(l, 1) if isinstance(l, str) else int(l) for l in predicted_labels]
                                report_text = classification_report(
                                    y_true_num, y_pred_num,
                                    target_names=["Negative", "Neutral", "Positive"],
                                    output_dict=False
                                )
                                st.code(report_text, language=None)
                            else:
                                st.info("Classification report requires sklearn library.")
                        except Exception as e:
                            st.warning(f"Could not generate classification report: {e}")
                
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

# ----------- Mode 5: Manual Text Input -----------
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