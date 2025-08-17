DiaBERT: Combating Diabetes Misinformation Using Transformer-Based Models
Project Overview

DiaBERT is an end-to-end misinformation detection system tailored to diabetes-related content.
Built on the BioBERT transformer model and enhanced through Domain-Adversarial Neural Networks (DANN), DiaBERT classifies online health claims into:

True

False

Partially True

It is deployed as a Chrome Extension that provides real-time credibility classification and explanations for users encountering health-related content online.


Key Features

Transformer Backbone: BioBERT (pre-trained on biomedical corpora)

Domain Adaptation: DANN to adapt from formal (medical) to informal (social media) domains

Three-Class Classification: True, False, Partially True

Content Filtering: SBERT + cosine similarity to filter only diabetes-related input

Explainability: Token-level highlights via Transformers Interpret (attention + integrated gradients)

Deployment: Real-time Chrome Extension calling an ONNX-optimized BioBERT model hosted on Fly.io



Dataset

Formal Dataset (DETERRENT):

2,269 diabetes-related claims

True: 1,661 | False: 608

Informal Dataset (social media: Facebook, Twitter/X, Reddit):

902 diabetes-related claims

True: 520 | False: 213 | Partially True: 163

Manually annotated and preprocessed (normalization, contraction expansion, emoji/URL filtering, unicode correction)



Model Pipeline

Stage 1: Supervised fine-tuning of BioBERT on formal two-class data

Stage 2: Domain adaptation via DANN — encoder learns domain-invariant features

Stage 3: Supervised fine-tuning on informal three-class dataset



Content Filtering (SBERT + Cosine Similarity)

Model: all-MiniLM-L6-v2

Diabetes domain embedding vector created from domain corpus

User input must exceed cosine similarity > 0.7 to be considered “in-domain”



Explainability

Explored: LIME, SHAP, Transformers Interpret

Chosen: Transformers Interpret

Highlights tokens driving the prediction

Works seamlessly with transformers + subword tokenization

Provides attention saliency + integrated gradients



Deployment
Backend (Flask API on Fly.io)

Endpoint: /predict → returns classification + explanation

Health check: /ping

SBERT filtering + explanation generation (template/GPT-based)

ONNX optimization for faster inference



Frontend (Chrome Extension)

Users can:

Manually enter text into the extension popup.

Highlight text on any webpage → when the popup is opened, the highlighted snippet is automatically prefilled.

Both methods allow the user to hit “Classify” and instantly get:

Prediction (True / False / Partially True)

Token-level highlights

Human-readable explanation

Sample Use Case

Blog post claims:

“Bitter leaf cures diabetes completely.”

DiaBERT output: False

Token highlights: “cures”, “completely”

Explanation:
“The claim suggests a definitive cure without scientific support. Bitter leaf may help regulate blood sugar but is not a standalone treatment.”



Resources

Extension (Chrome Web Store): https://chromewebstore.google.com/detail/diabert-classifier/pkccflhgplpbmoglflfjhhlnpdjbblpk?authuser=0&hl=en-GB

Code, datasets, and training scripts:https://github.com/PASCL-Lab/DiaBERT



Future Work

Expand dataset to multilingual claims (non-English)

Extend DiaBERT to other chronic illnesses (asthma, hypertension)

Evaluate demographic bias and add user-centric explanation toggles
