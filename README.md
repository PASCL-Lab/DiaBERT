# DiaBERT

> Domain-adapted transformer pipeline for detecting diabetes-related health misinformation, delivered as a real-time Chrome extension.

## Overview

DiaBERT is an end-to-end misinformation-detection system specialised for diabetes-related claims encountered on the open web. It is built on the BioBERT transformer, fine-tuned in stages on both formal medical claims (derived from DETERRENT) and informal, socially sourced claims (Facebook, X, Reddit), and bridged between the two distributions using a Domain-Adversarial Neural Network (DANN). The deployed model produces a three-class verdict — **True**, **False**, or **Partially True** — restricted to in-domain text by an SBERT cosine-similarity gate. Predictions are accompanied by token-level attributions from Transformers Interpret so that users see which terms drove the decision. The repository contains the full training and evaluation notebooks, the SBERT content filter, the ONNX-quantised production model, a Flask backend deployable to Fly.io, the published Chrome extension, the curated datasets, and the user-evaluation survey artefacts.

## Research Context

DiaBERT investigates whether biomedical transformers fine-tuned on formal medical literature can be adapted, via adversarial domain alignment, to the noisy register of social-media health claims. The companion documentation lives in `DiaBERT_ Readme.docx`; this repository accompanies that write-up with all code, datasets, and evaluation material.

## Features

- Three-class classification (True / False / Partially True) of short diabetes-related claims
- BioBERT backbone fine-tuned in three stages with DANN-based domain adaptation
- SBERT (`all-MiniLM-L6-v2`) + cosine-similarity gate (threshold > 0.7) to reject out-of-domain input
- Token-level explanations via Transformers Interpret (integrated gradients) compatible with subword tokenisation
- ONNX-exported model for low-latency CPU inference behind a Flask API
- Production Chrome extension (published on the Chrome Web Store) for real-time, in-browser analysis
- User-evaluation survey data and analysis notebook

## Architecture

The classifier follows a three-stage training pipeline:

1. **Supervised fine-tuning** of BioBERT on the formal DETERRENT-derived two-class corpus (1,661 True / 608 False).
2. **Domain adaptation** with a DANN head so the encoder learns features invariant between the formal and informal domains.
3. **Final fine-tuning** on the informal three-class corpus (575 True / 167 False / 160 Partially True) curated from Facebook, X, and Reddit and preprocessed with unicode normalisation, contraction expansion, and emoji/URL filtering.

At inference time the input is first embedded by SBERT and compared against an averaged diabetes-domain embedding; only queries passing the similarity threshold are routed to the BioBERT classifier (exported to ONNX and served via ONNX Runtime). Integrated-gradients attributions are computed on the PyTorch checkpoint and surfaced alongside the class probabilities. Explainability alternatives (LIME, SHAP) were evaluated during development; Transformers Interpret was selected for its alignment with the transformer attention structure.

## Tech Stack

- **Modeling:** PyTorch, Hugging Face Transformers (BioBERT), sentence-transformers, Transformers Interpret
- **Inference / Optimisation:** ONNX, ONNX Runtime
- **Backend:** Flask, Flask-CORS, OpenAI SDK (optional explanation generation)
- **Frontend:** Chrome Extension (Manifest V3) with Chart.js visualisations
- **Deployment:** Docker, Fly.io

## Getting Started

### Prerequisites

- Python 3.10+
- Google Chrome (for the extension)
- Docker (for the containerised backend)
- Optional: Fly.io CLI (`flyctl`) for cloud deployment
- Optional: OpenAI API key (if natural-language explanations are enabled)

### Installation

```bash
git clone https://github.com/PASCL-Lab/DiaBERT.git
cd DiaBERT/DiaBERT_Backend
pip install -r requirements.txt
```

The backend expects the following artefacts alongside `app.py`:

- `newbiobert_finetuned_3class.onnx` — quantised classifier
- `combined_embeddings.pt`, `combined_texts.pt` — SBERT domain reference
- `../newbiobert_model_3class/newbiobert_model_3class/pytorch_model.bin` — PyTorch checkpoint for attributions

### Running

#### Inference / API (local)

```bash
cd DiaBERT_Backend
flask --app app run --host 0.0.0.0 --port 8080
```

Endpoints:
- `POST /predict` — classify a claim and return probabilities + token attributions
- `GET /ping` — health check

#### Inference / API (Docker)

```bash
cd DiaBERT_Backend
docker build -t diabert-backend .
docker run --rm -p 8080:8080 diabert-backend
```

#### Deploy to Fly.io

```bash
cd DiaBERT_Backend
flyctl auth login
flyctl launch --no-deploy            # first time only
flyctl secrets set OPENAI_API_KEY=sk-...   # optional
flyctl deploy
```

#### Training

Training, ablation, and evaluation experiments are provided as notebooks under `Training and evaluation scripts/`:

```bash
jupyter notebook "Training and evaluation scripts/"
# Key notebooks:
#   Final Training pipeline(DiaBERT).ipynb
#   BioBERT +DANN.ipynb
#   BioBERT+ CORAL.ipynb
#   Experimentation pipeline with BioBERT, Bilstm and BERT.ipynb
#   SBERT plus cosine similarity.ipynb
#   Using transformer interprete for explainability.ipynb
```

#### Browser Extension

1. Start the backend (locally or via Fly.io) and update the request URL in `DiaBERT_Chrome/popup.js` if needed.
2. Open `chrome://extensions/` in Chrome.
3. Enable **Developer mode** (top-right toggle).
4. Click **Load unpacked** and select the `DiaBERT_Chrome/` directory.
5. The DiaBERT icon will appear in the toolbar.

The released build is available on the Chrome Web Store: https://chromewebstore.google.com/detail/diabert-classifier/pkccflhgplpbmoglflfjhhlnpdjbblpk

## Project Structure

```
DiaBERT_Backend/                                # Flask + ONNX inference service
  app.py
  Dockerfile
  fly.toml
  requirements.txt
DiaBERT_Chrome/                                 # Manifest V3 Chrome extension
  manifest.json
  popup.html / popup.js
  background.js
  content-script.js
  chart.js
  icons/
newbiobert_model_3class/                        # Fine-tuned BioBERT PyTorch checkpoint
Datasets/
  Diabetes_cleaned.csv                          # Formal corpus (DETERRENT-derived)
  Corrected_Labeled(Informal dataset).csv       # Informal three-class corpus
Training and evaluation scripts/                # Training, DANN/CORAL, and explainability notebooks
DiaBERT_Extension_Evaluation/                   # User-study survey, responses, and analysis
DiaBERT_ Readme.docx                            # Detailed methodology write-up
```

## License

This project is the intellectual property of **PASCL Lab**. All rights reserved.

Unauthorized copying, distribution, modification, or use of this codebase, in whole or in part, is strictly prohibited without prior written permission from PASCL Lab.

(c) 2026 PASCL Lab. All rights reserved.
