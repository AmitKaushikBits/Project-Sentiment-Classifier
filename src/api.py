import csv
import json
import re
import time
from pathlib import Path
from typing import List, Optional

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, field_validator

MODEL_PATH = "models/best_model.joblib"
VECTORIZER_PATH = "models/tfidf_vectorizer.joblib"
BEST_MODEL_META = "models/best_model.json"
PREDICTIONS_LOG = "logs/predictions_log.csv"

MAX_TEXT_LEN = 5000

URL_RE = re.compile(r"https?://\S+|www\.\S+")
HTML_RE = re.compile(r"<[^>]+>")
NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
MULTISPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    t = text.lower()
    t = URL_RE.sub(" ", t)
    t = HTML_RE.sub(" ", t)
    t = NON_ALNUM_RE.sub(" ", t)
    t = MULTISPACE_RE.sub(" ", t).strip()
    return t


app = FastAPI(
    title="Ticket/Review Sentiment Classifier API",
    description="Serves the best tracked model (see /model-info) for 3-class "
    "sentiment classification of support tickets / product reviews.",
    version="1.0.0",
)

PRODUCT_CATALOG = {
    "Mobile": [
        "Battery life is excellent and the camera quality is amazing.",
        "The screen is crisp, but the phone gets hot during gaming.",
        "I love the performance, but the price feels a little high.",
    ],
    "Headphone": [
        "Sound quality is fantastic and very comfortable to wear.",
        "The noise cancellation is decent, but the battery drains fast.",
        "Great product for travel, but the case feels a bit flimsy.",
    ],
    "iPad": [
        "The display is gorgeous and the tablet is super fast.",
        "I like the screen, but the accessories are too expensive.",
        "Very smooth for work and entertainment, highly recommended.",
    ],
    "Laptop": [
        "The keyboard feels great and the laptop starts up quickly.",
        "The fan is noisy when I run several applications at once.",
        "Good performance for work, but the charger is heavier than expected.",
    ],
    "Smartwatch": [
        "The fitness tracking is accurate and the display is easy to read.",
        "Notifications are useful, but the battery lasts only one day.",
        "Comfortable to wear and reasonably priced for the features.",
    ],
    "Camera": [
        "Photos look sharp and the autofocus works very well.",
        "The controls are confusing and the battery runs out too quickly.",
        "A solid camera for beginners with good image quality.",
    ],
    "Coffee Maker": [
        "It brews quickly and the coffee tastes rich every morning.",
        "The water reservoir is difficult to remove and clean.",
        "Simple controls and consistent results for the price.",
    ],
}

_state = {"model": None, "vectorizer": None, "meta": None}


@app.on_event("startup")
def load_artifacts():
    if not Path(MODEL_PATH).exists() or not Path(VECTORIZER_PATH).exists():
        # Fail loudly at startup rather than on first request
        raise RuntimeError(
            f"Model artifacts not found. Run the training pipeline first: "
            f"python src/ingest.py && python src/features.py && python src/train.py"
        )
    _state["model"] = joblib.load(MODEL_PATH)
    _state["vectorizer"] = joblib.load(VECTORIZER_PATH)
    if Path(BEST_MODEL_META).exists():
        _state["meta"] = json.loads(Path(BEST_MODEL_META).read_text())
    Path(PREDICTIONS_LOG).parent.mkdir(parents=True, exist_ok=True)
    if not Path(PREDICTIONS_LOG).exists():
        with open(PREDICTIONS_LOG, "w", newline="") as f:
            csv.writer(f).writerow(
                [
                    "timestamp",
                    "text",
                    "clean_text",
                    "predicted_label",
                    "confidence",
                    "n_tokens",
                    "is_edge_case",
                ]
            )


class PredictRequest(BaseModel):
    text: str = Field(..., description="Raw ticket/review text")

    @field_validator("text")
    @classmethod
    def not_none(cls, v):
        if v is None:
            raise ValueError("text must not be null")
        return v


class BatchPredictRequest(BaseModel):
    texts: List[str]


class PredictResponse(BaseModel):
    label: str
    confidence: Optional[float]
    is_edge_case: bool
    message: Optional[str] = None


def _score(text: str):
    """Returns (label, confidence, is_edge_case, message)."""
    if not isinstance(text, str):
        return "neutral", None, True, "Non-string input coerced to empty text."

    raw = text.strip()
    if len(raw) == 0:
        return (
            "neutral",
            None,
            True,
            "Empty/whitespace-only text; defaulted to 'neutral'.",
        )

    if len(raw) > MAX_TEXT_LEN:
        raw = raw[:MAX_TEXT_LEN]
        truncated_msg = f"Input truncated to {MAX_TEXT_LEN} characters."
    else:
        truncated_msg = None

    ctext = clean_text(raw)
    if len(ctext.split()) == 0:
        return (
            "neutral",
            None,
            True,
            "Text contained no usable tokens after cleaning; defaulted to 'neutral'.",
        )

    vectorizer = _state["vectorizer"]
    model = _state["model"]
    X = vectorizer.transform([ctext])

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        idx = probs.argmax()
        label = model.classes_[idx]
        confidence = float(probs[idx])
    else:
        # LinearSVC has no predict_proba; use decision_function margin
        # as a confidence proxy (min-max normalized across classes).
        label = model.predict(X)[0]
        try:
            scores = model.decision_function(X)[0]
            s = scores - scores.min()
            confidence = float(s.max() / (s.sum() + 1e-9)) if s.sum() > 0 else None
        except Exception:
            confidence = None

    return label, confidence, False, truncated_msg


# Logging function for predictions
def _log_prediction(text, clean, label, confidence, is_edge):
    with open(PREDICTIONS_LOG, "a", newline="") as f:
        csv.writer(f).writerow(
            [
                time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                text.replace("\n", " ")[:500] if isinstance(text, str) else "",
                clean[:500],
                label,
                confidence,
                len(clean.split()),
                is_edge,
            ]
        )


@app.get("/", response_class=HTMLResponse)
def root():
    return HTMLResponse("""
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Product Review Sentiment Dashboard</title>
    <style>
      :root {
        --bg: #07111f;
        --panel: #0f172a;
        --panel-alt: #111827;
        --card: #172033;
        --text: #e5e7eb;
        --muted: #94a3b8;
        --positive: #22c55e;
        --negative: #ef4444;
        --neutral: #f59e0b;
        --accent: #60a5fa;
        --border: rgba(148, 163, 184, 0.2);
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        font-family: Arial, sans-serif;
        background: linear-gradient(135deg, #07111f, #111827);
        color: var(--text);
        padding: 32px 16px;
      }
      .container {
        max-width: 1100px;
        margin: 0 auto;
        background: rgba(15, 23, 42, 0.9);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 20px 45px rgba(0,0,0,0.25);
      }
      h1 {
        margin: 0 0 10px;
        font-size: clamp(2rem, 4vw, 3rem);
      }
      .subtitle {
        margin: 0 0 18px;
        color: var(--muted);
      }
      .grid {
        display: grid;
        grid-template-columns: 1.2fr 0.8fr;
        gap: 20px;
      }
      .panel {
        background: var(--panel-alt);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 18px;
      }
      textarea {
        width: 100%;
        min-height: 220px;
        resize: vertical;
        border: 1px solid var(--border);
        border-radius: 12px;
        background: rgba(15, 23, 42, 0.8);
        color: var(--text);
        padding: 16px;
        font-size: 1rem;
        line-height: 1.5;
      }
      .toolbar {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        margin-top: 16px;
      }
      button {
        border: none;
        border-radius: 10px;
        cursor: pointer;
        padding: 12px 18px;
        font-weight: 700;
      }
      .primary {
        background: var(--accent);
        color: #06121d;
      }
      .secondary {
        background: rgba(96, 165, 250, 0.14);
        color: var(--text);
        border: 1px solid rgba(96, 165, 250, 0.35);
      }
      .summary-box {
        margin-top: 18px;
        padding: 16px;
        border-radius: 12px;
        background: rgba(96, 165, 250, 0.08);
        border: 1px solid rgba(96, 165, 250, 0.15);
      }
      .summary-box h3 {
        margin: 0 0 12px;
      }
      .counts {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 10px;
      }
      .count {
        padding: 12px 10px;
        border-radius: 10px;
        text-align: center;
        background: rgba(148, 163, 184, 0.08);
      }
      .count strong {
        display: block;
        font-size: 1.4rem;
        margin-top: 6px;
      }
      .count.positive strong { color: var(--positive); }
      .count.negative strong { color: var(--negative); }
      .count.neutral strong { color: var(--neutral); }
      .review-list {
        margin-top: 22px;
      }
      .review-item {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 14px;
        margin-top: 10px;
      }
      .review-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 12px;
        margin-bottom: 8px;
      }
      .badge {
        display: inline-block;
        border-radius: 999px;
        padding: 7px 10px;
        font-size: 0.78rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.05em;
      }
      .badge.positive { background: rgba(34, 197, 94, 0.18); color: #bbf7d0; }
      .badge.negative { background: rgba(239, 68, 68, 0.18); color: #fecaca; }
      .badge.neutral { background: rgba(245, 158, 11, 0.18); color: #fcd34d; }
      .review-item p {
        margin: 0;
        line-height: 1.6;
        color: #dfe7f4;
      }
      .confidence {
        color: var(--muted);
        font-size: 0.82rem;
        margin-top: 8px;
      }
      .empty-state {
        color: var(--muted);
        margin-top: 14px;
      }
      @media (max-width: 860px) {
        .grid { grid-template-columns: 1fr; }
      }
    </style>
  </head>
  <body>
    <main class="container">
      <h1>Sentiment Classifier</h1>
      <p class="subtitle">Product Review Dashboard — paste customer reviews and classify each one by sentiment.</p>
      <div class="products-inline" style="margin: 0 0 18px; color: var(--muted);">
        Products: Mobile, Headphone, iPad, Laptop, Smartwatch, Camera, Coffee Maker
      </div>

      <div class="grid">
        <section class="panel">
          <label for="productSelect" style="display:block; margin-bottom: 8px; font-weight: 700;">Select product</label>
          <select id="productSelect" style="width: 100%; padding: 12px; border-radius: 10px; border: 1px solid var(--border); background: rgba(15, 23, 42, 0.8); color: var(--text); margin-bottom: 16px;">
            <option value="">Choose a product</option>
          </select>

          <textarea id="reviewsInput" placeholder="Enter one review per line...\nExample:\nI love this product and it works perfectly.\nThis broke after one day and I am very disappointed.\nThe setup was easy and the quality is great.\n">I love this product and it works perfectly.\nThis broke after one day and I am very disappointed.\nThe setup was easy and the quality is great.</textarea>

          <div class="toolbar">
            <button class="primary" id="analyzeBtn" type="button">Analyze reviews</button>
            <button class="secondary" id="loadProductBtn" type="button">Load selected product</button>
            <button class="secondary" id="sampleBtn" type="button">Load sample</button>
          </div>
        </section>

        <aside class="panel">
          <div class="summary-box">
            <h3>Sentiment Summary</h3>
            <div class="counts">
              <div class="count positive">
                Positive
                <strong id="positiveCount">0</strong>
              </div>
              <div class="count neutral">
                Neutral
                <strong id="neutralCount">0</strong>
              </div>
              <div class="count negative">
                Negative
                <strong id="negativeCount">0</strong>
              </div>
            </div>
          </div>
        </aside>
      </div>

      <section class="review-list" id="reviewList">
        <div class="empty-state">No reviews analyzed yet.</div>
      </section>
    </main>

    <script>
      const reviewsInput = document.getElementById('reviewsInput');
      const productSelect = document.getElementById('productSelect');
      const analyzeBtn = document.getElementById('analyzeBtn');
      const loadProductBtn = document.getElementById('loadProductBtn');
      const sampleBtn = document.getElementById('sampleBtn');
      const reviewList = document.getElementById('reviewList');
      const positiveCount = document.getElementById('positiveCount');
      const neutralCount = document.getElementById('neutralCount');
      const negativeCount = document.getElementById('negativeCount');

      const sampleReviews = [
        'I love this product and it works perfectly.',
        'The battery drains too fast and the screen is poor.',
        'It is okay, but I expected better quality for the price.'
      ];
      const newline = String.fromCharCode(10);

      async function loadProductOptions() {
        try {
          const response = await fetch('/products');
          const data = await response.json();
          const products = data.products || [];
          productSelect.innerHTML = '<option value="">Choose a product</option>' +
            products.map(product => `<option value="${product.name}">${product.name}</option>`).join('');
          if (products.length) {
            productSelect.value = products[0].name;
            reviewsInput.value = products[0].reviews.join(newline);
          }
        } catch (error) {
          console.error('Failed to load products', error);
        }
      }

      function setSummary(results) {
        const counts = { positive: 0, neutral: 0, negative: 0 };
        for (const item of results) {
          const label = (item.label || 'neutral').toLowerCase();
          if (label in counts) counts[label] += 1;
        }
        positiveCount.textContent = counts.positive;
        neutralCount.textContent = counts.neutral;
        negativeCount.textContent = counts.negative;
      }

      function renderResults(results) {
        reviewList.innerHTML = '';
        if (!results.length) {
          reviewList.innerHTML = '<div class="empty-state">No reviews analyzed yet.</div>';
          setSummary([]);
          return;
        }

        setSummary(results);
        results.forEach((item, index) => {
          const row = document.createElement('div');
          row.className = 'review-item';
          const label = (item.label || 'neutral').toLowerCase();
          const confidence = typeof item.confidence === 'number'
            ? `Confidence: ${(item.confidence * 100).toFixed(1)}%`
            : 'Confidence: n/a';
          row.innerHTML = `
            <div class="review-top">
              <strong>${productSelect.value || 'Review'} ${index + 1}</strong>
              <span class="badge ${label}">${item.label || 'Neutral'}</span>
            </div>
            <p>${(item.text || '').replace(/</g, '&lt;').replace(/>/g, '&gt;')}</p>
            <div class="confidence">${confidence}</div>
          `;
          reviewList.appendChild(row);
        });
      }

      async function loadSelectedProduct() {
        const selected = productSelect.value;
        if (!selected) {
          reviewList.innerHTML = '<div class="empty-state">Please select a product first.</div>';
          return;
        }

        try {
          const response = await fetch('/products');
          const data = await response.json();
          const product = (data.products || []).find(item => item.name === selected);
          if (!product) {
            reviewList.innerHTML = '<div class="empty-state">Selected product not found.</div>';
            return;
          }
          reviewsInput.value = product.reviews.join(newline);
          reviewList.innerHTML = '<div class="empty-state">Loaded product reviews.</div>';
        } catch (error) {
          reviewList.innerHTML = '<div class="empty-state">Unable to load selected product reviews.</div>';
        }
      }

      async function analyzeReviews() {
        const raw = reviewsInput.value;
        const texts = raw
          .split(newline)
          .map(item => item.trim())
          .filter(Boolean);

        if (!texts.length) {
          renderResults([]);
          reviewList.innerHTML = '<div class="empty-state">Please enter at least one review.</div>';
          return;
        }

        reviewList.innerHTML = '<div class="empty-state">Analyzing reviews...</div>';

        try {
          const response = await fetch('/predict/batch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ texts })
          });

          const data = await response.json();
          if (!response.ok) {
            throw new Error(data.detail || 'Batch prediction failed.');
          }

          const results = (data.results || []).map((item, index) => ({
            ...item,
            text: texts[index] || ''
          }));

          renderResults(results);
        } catch (error) {
          reviewList.innerHTML = `<div class="empty-state">${error.message || 'Unable to classify reviews right now.'}</div>`;
        }
      }

      productSelect.addEventListener('change', () => {
        if (productSelect.value) {
          loadSelectedProduct();
        }
      });
      analyzeBtn.addEventListener('click', analyzeReviews);
      loadProductBtn.addEventListener('click', loadSelectedProduct);
      sampleBtn.addEventListener('click', () => {
        reviewsInput.value = sampleReviews.join(newline);
        analyzeReviews();
      });

      loadProductOptions();
    </script>
  </body>
</html>
    """)


@app.get("/products")
def get_products():
    products = []
    for name, reviews in PRODUCT_CATALOG.items():
        products.append({"name": name, "reviews": reviews})
    return {"products": products}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _state["model"] is not None}


@app.get("/model-info")
def model_info():
    if _state["meta"] is None:
        return {"model": "unknown", "metrics": {}}
    return _state["meta"]


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        label, confidence, is_edge, message = _score(req.text)
        ctext = clean_text(req.text) if isinstance(req.text, str) else ""
        _log_prediction(req.text, ctext, label, confidence, is_edge)
        return PredictResponse(
            label=label, confidence=confidence, is_edge_case=is_edge, message=message
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")


@app.post("/predict/batch")
def predict_batch(req: BatchPredictRequest):
    if len(req.texts) == 0:
        raise HTTPException(status_code=400, detail="texts list is empty")
    if len(req.texts) > 500:
        raise HTTPException(status_code=400, detail="Batch too large (max 500)")
    results = []
    for text in req.texts:
        label, confidence, is_edge, message = _score(text)
        ctext = clean_text(text) if isinstance(text, str) else ""
        _log_prediction(text, ctext, label, confidence, is_edge)
        results.append(
            {
                "label": label,
                "confidence": confidence,
                "is_edge_case": is_edge,
                "message": message,
            }
        )
    return {"results": results}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=False)
