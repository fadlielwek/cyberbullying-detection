import os
import re
import string
import pickle
import emoji


from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
import nltk

nltk.data.path.append("/tmp/nltk_data")
nltk.download('punkt', download_dir="/tmp/nltk_data")
nltk.download('stopwords', download_dir="/tmp/nltk_data")



# =========================
# PATH & LOAD MODEL
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

# =========================
# FASTAPI INIT
# =========================
app = FastAPI(
    title="Cyberbullying Comment Detection App",
    description="Pendeteksi Komentar Cyberbullying Instagram",
    version="1.0.0"
)

templates = Jinja2Templates(
    directory=os.path.join(BASE_DIR, "templates")
)

# =========================
# PREPROCESSING (SAMA PERSIS DENGAN TRAINING)
# =========================
stop_words = set(stopwords.words("indonesian"))

def clean_text(text: str) -> str:
    text = text.lower()

    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()

    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    text = emoji.demojize(text)
    text = text.encode("ascii", "ignore").decode("ascii")

    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]

    return " ".join(tokens)

# =========================
# SCHEMA
# =========================
class CommentRequest(BaseModel):
    email: str   # ini disesuaikan dengan frontend (email: comment)

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float

# =========================
# ROUTES
# =========================
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: CommentRequest):
    # preprocessing
    cleaned_text = clean_text(request.email)

    # vectorize
    X = vectorizer.transform([cleaned_text])

    # predict
    pred = model.predict(X)[0]
    prob = model.predict_proba(X).max()

    return {
        "prediction": "bullying" if pred == 1 else "non-bullying",
        "confidence": round(float(prob), 4)
    }

@app.get("/classify", response_class=HTMLResponse)
async def classify_page(request: Request):
    return templates.TemplateResponse(
        "predict.html",
        {"request": request}
    )

app.mount("/static", StaticFiles(directory="static"), name="static")
