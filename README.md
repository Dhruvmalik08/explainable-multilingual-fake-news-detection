# 📰 Explainable Multilingual Fake News Detection

A multimodal (text + image) fake news classifier with word-level explainability and
automated evidence retrieval, built as a minor project. The system classifies
English and Hindi news articles — with or without an accompanying image — as
**FAKE** or **REAL**, then shows *why* using SHAP, and cross-checks the claim
against Google's Fact Check Tools API.

## Features

* **Multimodal classification** — MuRIL (text) + ResNet-50 (image) features fused
through a late-fusion classifier, with dedicated calibrated thresholds for
text-only, image-only, and combined input.
* **Explainability** — word-level SHAP attributions (permutation explainer) show
which words pushed the prediction toward FAKE or REAL.
* **Evidence retrieval** — queries the Google Fact Check Tools API for prior
fact-checks matching the article.
* **Multilingual** — MuRIL supports Hindi natively; the app flags Hindi
predictions as exploratory since calibration was performed on English data.
* **Interactive demo** — Streamlit app (`app.py`) for live inference.

## Architecture

```
Raw text ──► MuRIL (google/muril-base-cased) ──► 768-dim embedding ─┐
                                                                      ├─► Fusion (concat) ──► Classifier ──► FAKE / REAL
Raw image ─► ResNet-50 (ImageNet pretrained) ──► 2048-dim embedding ─┘
```

* `text\\\\\\\\\\\\\\\_branch`: 768 → 512 → 256 (ReLU, Dropout)
* `image\\\\\\\\\\\\\\\_branch`: 2048 → 512 → 256 (ReLU, Dropout)
* `classifier`: 512 (concat) → 256 → 2

Per-mode decision thresholds (tuned on validation data, not a flat 0.5 cutoff):

|Mode|Threshold|
|-|-|
|Multimodal|0.48|
|Text-only|0.46|
|Image-only|0.56|

## Repo structure

```
.
├── app.py                     # Streamlit inference app
├── training\\\\\\\\\\\\\\\_pipeline.ipynb    # Feature extraction, training, calibration, SHAP, evidence retrieval
├── requirements.txt           # Dependencies to run app.py
├── requirements-training.txt  # Additional dependencies to run the notebook
├── .env.example                # Template for required API key
└── models/                    # NOT included in this repo — see "Model weights" below
```

## Dataset

Sourced from **FakeNewsNet** (multimodal subset), balanced to 10,000 / 2,000 / 2,000
train/val/test articles with paired images. The dataset is English-only; Hindi
coverage was added by machine-translating the English text (same images and
labels) rather than sourcing a separate Hindi corpus — so Hindi results should be
read as exploratory rather than fully validated.

Raw data is not included in this repo. See `training\\\\\\\\\\\\\\\_pipeline.ipynb` for the
expected directory layout under `data/`.

## Model weights



Trained checkpoints (`v2\_unified\_model\_best.pt`, `v2\_normalization\_stats.pt`) are

not committed here — they're too large for a plain git push. To run `app.py`

locally:

1. Train the model with training\_pipeline.ipynb, or download the pretrained
checkpoints from <https://drive.google.com/drive/folders/1QxRgZHs2xSXtjYoXdKpn-7FTihk8MOHV?usp=sharing>.

2\.  Place both files in a local `models/` directory (already git-ignored).

## Setup

```bash
git clone <this-repo-url>
cd <repo-name>
python -m venv venv \\\\\\\\\\\\\\\&\\\\\\\\\\\\\\\& source venv/bin/activate   # optional but recommended
pip install -r requirements.txt
cp .env.example .env   # then fill in FACT\\\\\\\\\\\\\\\_CHECK\\\\\\\\\\\\\\\_API\\\\\\\\\\\\\\\_KEY
```

Get a Fact Check API key from the
[Google Cloud Console](https://console.cloud.google.com/apis/library/factchecktools.googleapis.com).
**Never commit your real `.env` file or paste the key directly into source.**

## Running the app

```bash
streamlit run app.py
```

The app loads `models/v2\\\\\\\\\\\\\\\_unified\\\\\\\\\\\\\\\_model\\\\\\\\\\\\\\\_best.pt` and
`models/v2\\\\\\\\\\\\\\\_normalization\\\\\\\\\\\\\\\_stats.pt` on first run (cached after that), then lets
you paste article text, upload an image, or both, and returns a prediction with
SHAP word attributions and Fact Check evidence.

## Reproducing training

Open `training\\\\\\\\\\\\\\\_pipeline.ipynb`. It expects:

* MuRIL and ResNet-50 features pre-extracted and saved to `data/features/`
* A Google Drive-mounted layout (the notebook was developed in Colab) — adjust
`PROJECT\\\\\\\\\\\\\\\_ROOT` at the top if running elsewhere
* A Fact Check API key pasted into the designated cell (not committed)

Install the extra training dependencies first:

```bash
pip install -r requirements-training.txt
```

## Results

Final calibrated test-set performance (MuRIL + ResNet-50, per-mode thresholds
tuned on validation data):

|Mode|Threshold|Accuracy|Macro F1|FAKE Recall|REAL Recall|ROC-AUC|
|-|-|-|-|-|-|-|
|Multimodal|0.48|86.00%|0.8595|0.803|0.917|0.9281|
|Text-only|0.46|80.85%|0.8083|0.779|0.838|0.8898|
|Image-only|0.56|77.25%|0.7720|0.727|0.818|0.8446|

Multimodal fusion outperforms either single modality on every metric, confirming
that combining text and image signals adds real value rather than just
inheriting the stronger modality's performance.

* Explainability: SHAP modality-contribution analysis quantifies how much each
modality (text vs. image) drove a given prediction.

## Known limitations

* Hindi predictions are exploratory — the model was calibrated on English data
and Hindi text was obtained via machine translation, not native Hindi sources.
* The Fact Check API only returns matches for claims already reviewed by
professional fact-checkers, so most everyday articles will show no evidence —
this is expected, not a system failure.

## Author

Dhruv Malik — B.Tech CSE, Maharaja Surajmal Institute of Technology (GGSIPU)

## License

MIT — see `LICENSE`.

