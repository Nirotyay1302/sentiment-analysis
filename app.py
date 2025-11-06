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

# Try to import snscrape (optional - may not work on all platforms)
# Note: snscrape is archived and incompatible with Python 3.13+
# Default to False - will be set to True only if import succeeds
SNSCRAPE_AVAILABLE = False
sntwitter = None

# Only attempt import if not on Streamlit Cloud (Python 3.13+)
import sys
if sys.version_info < (3, 13):
    try:
        import snscrape.modules.twitter as sntwitter  # type: ignore
        SNSCRAPE_AVAILABLE = True
    except Exception:
        SNSCRAPE_AVAILABLE = False
        sntwitter = None
else:
    # Python 3.13+ - snscrape is known to be incompatible
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
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Try to import wordcloud and matplotlib for visualization
try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

# (OCR support removed for simpler hosting; app no longer requires Tesseract)

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
    if TRANSFORMERS_AVAILABLE:
        try:
            return TransformerSentimentModel()
        except Exception as e:
            # Don't use st.warning here as it may be called before Streamlit is initialized
            print(f"Transformer model not available: {e}. Using fallback model.")
            return None
    return None

# Initialize models lazily (transformer will be loaded on first use)
transformer_pipe = None
pipe = _try_load_model()

# Use transformer model if available, otherwise use joblib model
active_pipe = pipe

labels = {0: "Negative", 1: "Neutral", 2: "Positive"}

def ensure_model_ui():
    """In-UI helper: if no model loaded, prompt user to upload one."""
    global active_pipe, transformer_pipe, pipe
    
    # Try to load transformer model first (preferred, lazy loading)
    if TRANSFORMERS_AVAILABLE and transformer_pipe is None:
        try:
            with st.spinner("Loading transformer model (first time may take a moment to download ~500MB)..."):
                transformer_pipe = _load_transformer_model()
                if transformer_pipe is not None:
                    active_pipe = transformer_pipe
                    st.success("✅ Transformer model loaded successfully!")
                    return True
        except Exception as e:
            st.warning(f"Could not load transformer model: {e}")
    
    # If transformer model is available, we're good
    if transformer_pipe is not None:
        active_pipe = transformer_pipe
        return True
    
    # If joblib model is available, use it
    if pipe is not None:
        active_pipe = pipe
        return True
    
    # No model available - offer to upload joblib model
    st.warning("No model found. Using transformer model (if available) or upload a trained model (model.joblib) below.")
    
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
    
    return False

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
    st.sidebar.warning("⚠️ No Model Loaded")
    st.sidebar.caption("Loading transformer model or upload 'model.joblib'")
st.markdown("<h1 style='text-align: center; color: #0066cc; margin-bottom: 0.5em;'>📊 Social Media Sentiment Analyzer</h1>", unsafe_allow_html=True)

def clean_text(text):
    text = re.sub(r'[^A-Za-z0-9\s.,!?]', '', text)
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
            # Heuristic: prefer a column with 'text' in the name; otherwise pick the column with the highest average token length
            cols = list(df.columns)
            detected_col = None
            for c in cols:
                if 'text' in str(c).lower():
                    detected_col = c
                    break
            if detected_col is None:
                # score columns by average length of string values (ignoring empty cells)
                scores = {}
                for c in cols:
                    try:
                        vals = df[c].astype(str).replace('nan','').tolist()
                        lengths = [len(re.sub(r'[^A-Za-z0-9\s]', '', v).strip()) for v in vals if v and v.strip()]
                        scores[c] = sum(lengths)/len(lengths) if lengths else 0
                    except Exception:
                        scores[c] = 0
                detected_col = max(scores, key=scores.get)  # type: ignore

            text_col = st.selectbox("Select text column for sentiment analysis", options=cols, index=cols.index(detected_col))
            # warn if the selected column looks mostly numeric or very short
            sample_vals = df[text_col].astype(str).head(20).tolist()
            num_numeric = sum(1 for v in sample_vals if re.fullmatch(r"\s*\d+\s*", v))
            if num_numeric > 5:
                st.warning(f"The selected column '{text_col}' appears to contain many numeric or index-like values. If this is incorrect, pick a different column.")
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
                
                # Ensure model is loaded
                if active_pipe is None:
                    st.error("Model not loaded. Please ensure a model is available.")
                    st.stop()
                
                # Add progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Process in chunks with progress updates
                chunk_size = max(1, len(texts) // 50)  # Update 50 times
                preds = []
                for i in range(0, len(texts), chunk_size):
                    chunk = texts[i:i+chunk_size]
                    chunk_preds = active_pipe.predict(chunk)  # type: ignore
                    preds.extend(chunk_preds)
                    
                    # Update progress
                    progress = min(1.0, (i + len(chunk)) / len(texts))
                    progress_bar.progress(progress)
                    status_text.text(f"Processed {len(preds)}/{len(texts)} texts...")
                
                progress_bar.empty()
                status_text.empty()
                
                df["Predicted"] = preds
                df["Predicted_Label"] = df["Predicted"].map(labels)  # type: ignore
                result_df = df
                st.success("Full dataset analysis complete")
            else:
                # Ensure model is loaded
                if active_pipe is None:
                    st.error("Model not loaded. Please ensure a model is available.")
                    st.stop()
                
                # Use first N rows (deterministic) instead of random sampling
                head_df = df.head(int(sample_size)).copy()
                head_texts = head_df[text_col].astype(str).apply(clean_text).tolist()
                head_preds = active_pipe.predict(head_texts)  # type: ignore
                head_df["Predicted"] = head_preds
                head_df["Predicted_Label"] = head_df["Predicted"].map(labels)  # type: ignore
                result_df = head_df
                st.success(f"Quick scan (first {sample_size} rows) complete")

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
    # Show warnings if libraries aren't available
    if not SNSCRAPE_AVAILABLE:
        st.warning("⚠️ Twitter scraping is not available. The `snscrape` library may not be compatible with this platform.")
    if not YOUTUBE_DL_AVAILABLE:
        st.warning("⚠️ YouTube comment scraping is not available. The `youtube-comment-downloader` library may not be installed.")
    
    link = st.text_input("Paste a Twitter or YouTube link:")
    if st.button("Fetch & Analyze"):
        # Ensure model is available before analysis
        if not ensure_model_ui():
            st.stop()

        comments = []

        # Twitter support
        if "twitter.com" in link:
            if not SNSCRAPE_AVAILABLE:
                st.error("Twitter scraping is not available on this platform. Please use 'Analyze Dataset' or 'Manual Text Input' modes instead.")
                st.stop()
            comments = fetch_twitter_replies(link, limit=100)

        # YouTube support
        elif "youtube.com" in link or "youtu.be" in link:
            if not YOUTUBE_DL_AVAILABLE:
                st.error("YouTube comment scraping is not available. Please use 'Analyze Dataset' or 'Manual Text Input' modes instead.")
                st.stop()
            comments = fetch_youtube_comments(link, limit=100)

        if comments:
            cleaned_comments = [clean_text(c) for c in comments]
            
            # Ensure model is loaded
            if active_pipe is None:
                st.error("Model not loaded. Please ensure a model is available.")
                st.stop()
            
            # Add progress bar for social media analysis
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process with progress updates
            chunk_size = max(1, len(cleaned_comments) // 20)  # Update 20 times
            preds = []
            for i in range(0, len(cleaned_comments), chunk_size):
                chunk = cleaned_comments[i:i+chunk_size]
                chunk_preds = active_pipe.predict(chunk)  # type: ignore
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

# (Screenshot / OCR mode removed for simplified hosting)

# ----------- Mode 4: Manual Text Input -----------
elif mode == "Manual Text Input":
    # Ensure model is available before analysis
    if not ensure_model_ui():
        st.stop()

    text = st.text_area("Enter text:")
    if st.button("Analyze Text"):
        if text.strip():
            # Ensure model is loaded
            if active_pipe is None:
                st.error("Model not loaded. Please ensure a model is available.")
                st.stop()
            
            cleaned = clean_text(text)
            
            # Get predictions and probabilities
            if hasattr(active_pipe, 'predict_proba'):
                probas = active_pipe.predict_proba([cleaned])[0]  # type: ignore
            else:
                # Fallback for transformer model if predict_proba format differs
                if transformer_pipe is not None:
                    probas = transformer_pipe.predict_proba([cleaned])[0]
                else:
                    # For models without predict_proba, use predict and estimate confidence
                    pred = active_pipe.predict([cleaned])[0]  # type: ignore
                    probas = np.array([0.33, 0.33, 0.34])  # Default uniform
                    probas[pred] = 0.8  # Assign higher confidence to predicted class
            
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