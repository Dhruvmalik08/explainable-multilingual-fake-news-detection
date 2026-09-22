import os
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import re
import requests
from pathlib import Path
from PIL import Image
from transformers import AutoTokenizer, AutoModel
from torchvision import models
import shap

st.set_page_config(page_title="Explainable Multilingual Fake News Detection", layout="wide")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

V2_MODELS = Path("models")

CLASS_NAMES = ["FAKE", "REAL"]
THRESHOLDS = {"multimodal": 0.48, "text_only": 0.46, "image_only": 0.56}

FACT_CHECK_API_KEY = os.environ.get("FACT_CHECK_API_KEY", "")
if not FACT_CHECK_API_KEY:
    st.warning(
        "No Fact Check API key found. Set the FACT_CHECK_API_KEY environment variable "
        "(see .env.example) to enable evidence retrieval."
    )

class UnifiedMultimodalClassifier(nn.Module):
    def __init__(self, text_dim=768, image_dim=2048, hidden_dim=256, num_classes=2, dropout=0.4):
        super().__init__()
        self.text_branch = nn.Sequential(nn.Linear(text_dim,512), nn.ReLU(), nn.Dropout(dropout), nn.Linear(512,hidden_dim), nn.ReLU())
        self.image_branch = nn.Sequential(nn.Linear(image_dim,512), nn.ReLU(), nn.Dropout(dropout), nn.Linear(512,hidden_dim), nn.ReLU())
        self.classifier = nn.Sequential(nn.Linear(hidden_dim*2,256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256,num_classes))
    def forward(self, t, i):
        return self.classifier(torch.cat([self.text_branch(t), self.image_branch(i)], dim=1))

@st.cache_resource
def load_everything():
    norm = torch.load(V2_MODELS / "v2_normalization_stats.pt", map_location=DEVICE)
    text_mean, text_std = norm["text_mean"].to(DEVICE), norm["text_std"].to(DEVICE)
    image_mean, image_std = norm["image_mean"].to(DEVICE), norm["image_std"].to(DEVICE)

    model = UnifiedMultimodalClassifier().to(DEVICE)
    ckpt = torch.load(V2_MODELS / "v2_unified_model_best.pt", map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")
    text_model = AutoModel.from_pretrained("google/muril-base-cased").to(DEVICE)
    text_model.eval()

    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    image_encoder = nn.Sequential(*list(resnet.children())[:-1]).to(DEVICE)
    image_encoder.eval()
    image_transform = models.ResNet50_Weights.DEFAULT.transforms()

    masker = shap.maskers.Text(tokenizer=r"\W+")
    return model, tokenizer, text_model, image_encoder, image_transform, masker, text_mean, text_std, image_mean, image_std

model, tokenizer, text_model, image_encoder, image_transform, masker, text_mean, text_std, image_mean, image_std = load_everything()

@torch.no_grad()
def encode_text(text):
    encoded = tokenizer([text], padding=True, truncation=True, max_length=256, return_tensors="pt").to(DEVICE)
    out = text_model(**encoded)
    mask = encoded["attention_mask"].unsqueeze(-1).expand(out.last_hidden_state.size()).float()
    emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    return (emb - text_mean) / text_std

@torch.no_grad()
def encode_image(pil_image):
    img = image_transform(pil_image.convert("RGB")).unsqueeze(0).to(DEVICE)
    feat = image_encoder(img).flatten(1)
    return (feat - image_mean) / image_std

@torch.no_grad()
def predict(text=None, pil_image=None):
    text_vec = encode_text(text) if text else torch.zeros(1, 768).to(DEVICE)
    image_vec = encode_image(pil_image) if pil_image is not None else torch.zeros(1, 2048).to(DEVICE)
    mode = "multimodal" if (text and pil_image is not None) else ("text_only" if text else "image_only")
    probs = torch.softmax(model(text_vec, image_vec), dim=1)[0]
    pred = 1 if probs[1].item() >= THRESHOLDS[mode] else 0
    return {"mode": mode, "prediction": CLASS_NAMES[pred], "confidence": probs[pred].item(),
            "fake_prob": probs[0].item(), "real_prob": probs[1].item()}

@torch.no_grad()
def predict_proba_from_text(texts, fixed_image_vec):
    texts = [str(t) if t else "" for t in texts]
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(DEVICE)
    out = text_model(**encoded)
    mask = encoded["attention_mask"].unsqueeze(-1).expand(out.last_hidden_state.size()).float()
    text_emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    text_emb = (text_emb - text_mean) / text_std
    img_batch = fixed_image_vec.repeat(len(texts), 1)
    probs = torch.softmax(model(text_emb, img_batch), dim=1)
    return probs.cpu().numpy()

def explain_text(text, image_vec, prediction):
    try:
        n_words = max(len(re.findall(r"\w+", text)), 1)
        max_evals = max(2 * n_words + 1, 500)
        explainer = shap.Explainer(lambda x: predict_proba_from_text(x, image_vec), masker,
                                    output_names=CLASS_NAMES, algorithm="permutation")
        shap_values = explainer([text], max_evals=max_evals)
        pred_idx = CLASS_NAMES.index(prediction)
        values = shap_values.values[0][:, pred_idx]
        return sorted(zip(shap_values.data[0], values), key=lambda x: -abs(x[1]))[:8]
    except Exception:
        return None

def fact_check_search(query, max_results=3):
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {"query": query[:200], "key": FACT_CHECK_API_KEY, "pageSize": max_results}
    try:
        resp = requests.get(url, params=params, timeout=15)
        status, raw_text = resp.status_code, resp.text
        resp.raise_for_status()
        claims = resp.json().get("claims", [])[:max_results]
    except requests.exceptions.RequestException as e:
        return {"ok": False, "error": str(e), "status": None, "raw": None, "results": []}
    out = [{"rating": (c.get("claimReview") or [{}])[0].get("textualRating", "N/A"),
            "publisher": (c.get("claimReview") or [{}])[0].get("publisher", {}).get("name", "Unknown"),
            "url": (c.get("claimReview") or [{}])[0].get("url", "")} for c in claims]
    return {"ok": True, "error": None, "status": status, "raw": raw_text, "results": out}

SAMPLE_ARTICLES = {
    "-- Select a verified sample --": "",
    "🇬🇧 Surprised owl (reliable, has evidence)": "surprised owl",
    "🇬🇧 Climate report (has fact-check match)": "report on climate change shows canada warming at twice the rate of rest of world",
    "🇬🇧 Rotavirus vaccine (has fact-check match)": "new vaccine to fight rotavirus a disease that kills children a day",
    "🇬🇧 Skull-shaped clouds (has fact-check match)": "the clouds in this pic i took looks almost like a skull",
    "🇮🇳 Hindi — Plastic rice (has fact-check match)": "चावल प्लास्टिक से बनाया जा रहा है",
    "🇮🇳 Hindi — Jio free recharge (has fact-check match)": "जियो दे रहा है मुफ्त में 1 साल का रिचार्ज",
}
def first_sentence(text, max_chars=200):
    match = re.match(r'^(.*?[.!?])\s', str(text) + ' ')
    sentence = match.group(1) if match else str(text)[:max_chars]
    return sentence[:max_chars]
# ---------------- UI ----------------
st.title("📰 Explainable Multilingual Fake News Detection")
st.caption("MuRIL classification · SHAP word-level explainability · Google Fact Check evidence")

lang = st.radio("Article language", ["English", "Hindi (हिन्दी)"], horizontal=True)
if lang.startswith("Hindi"):
    st.info("MuRIL supports Hindi text natively, but this system's calibration and evaluation were performed on English data — treat Hindi predictions as exploratory.")
col1, col2 = st.columns(2)
with col1:
    sample_choice = st.selectbox("Or load a verified sample", list(SAMPLE_ARTICLES.keys()))
    text_input = st.text_area("News article text", value=SAMPLE_ARTICLES[sample_choice],
                               height=200, placeholder="Paste a news article here...")
with col2:
    image_input = st.file_uploader("News image (optional)", type=["jpg", "jpeg", "png"])
    if image_input:
        st.image(image_input, caption="Uploaded image", use_container_width=True)

if st.button("Analyze", type="primary"):
    if not text_input and not image_input:
        st.warning("Please provide article text, an image, or both.")
    else:
        pil_img = Image.open(image_input) if image_input else None
        with st.spinner("Analyzing..."):
            result = predict(text=text_input if text_input else None, pil_image=pil_img)

        st.divider()
        badge = "🟢" if result["prediction"] == "REAL" else "🔴"
        st.subheader(f"{badge} Prediction: {result['prediction']}  ({result['confidence']:.1%} confidence)")
        st.caption(f"Mode used: {result['mode'].replace('_', ' ')}")

        c1, c2 = st.columns(2)
        c1.metric("FAKE probability", f"{result['fake_prob']:.1%}")
        c2.metric("REAL probability", f"{result['real_prob']:.1%}")

        if text_input:
            st.markdown("### 🔍 Word-level explanation (SHAP)")
            image_vec = encode_image(pil_img) if pil_img is not None else torch.zeros(1, 2048).to(DEVICE)
            top_words = explain_text(text_input, image_vec, result["prediction"])
            if top_words:
                for w, v in top_words:
                    color = "#2ecc71" if v > 0 else "#e74c3c"
                    direction = f"pushes toward {result['prediction']}" if v > 0 else f"pushes away from {result['prediction']}"
                    st.markdown(f"<div style='background:{color}22;border-left:4px solid {color};padding:6px 10px;margin:4px 0;'>"
                                f"<b>{w.strip()}</b> &nbsp; ({v:+.3f}) — {direction}</div>", unsafe_allow_html=True)
            else:
                st.info("Word-level explanation unavailable — text is too short to explain meaningfully.")

        st.markdown("### 📋 Fact Check evidence")
        if text_input:
            query_used = first_sentence(text_input)
            fc = fact_check_search(query_used)
            if not fc["ok"]:
                st.error(f"Fact Check API call failed: {fc['error']}")
            elif fc["results"]:
                for e in fc["results"]:
                    st.markdown(f"**[{e['rating']}]** {e['publisher']} — [{e['url']}]({e['url']})")
            else:
                st.caption("No matching fact-check records for this query — expected for most "
                        "articles, since Google's index only covers claims already reviewed "
                        "by professional fact-checkers.")
            with st.expander("🔧 Raw Fact Check API call (verification for grading)"):
                st.write(f"**Query sent:** `{query_used}`")
                st.write(f"**HTTP status:** {fc['status']}")
                if fc["raw"]:
                    st.code(fc["raw"], language="json")
        else:
            st.caption("Fact-check evidence requires article text.")