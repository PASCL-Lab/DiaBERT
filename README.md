# DiaBERT: Combating Diabetes Misinformation Using Transformer-Based Models

## Project Overview
DiaBERT is an end-to-end misinformation detection system tailored to diabetes-related content.  
Built on the BioBERT transformer model and enhanced through domain adaptation (DANN), DiaBERT classifies online health claims into **True, False, or Partially True**.  
It is deployed as a **Chrome extension** that provides real-time credibility classification and explanation for users encountering health-related content online.

---

## Key Features
- **Transformer Backbone**: Built on BioBERT (a BERT model pre-trained on biomedical corpora)  
- **Domain Adaptation**: Implemented DANN (Domain-Adversarial Neural Network) to adapt from formal (medical) to informal (social media) domains  
- **Three-Class Classification**: True, False, Partially True  
- **Content Filtering**: SBERT + Cosine Similarity to filter only diabetes-related input  
- **Explainability**: Worked with LIME, SHAP, and Transformers Interpret (final choice: Transformers Interpret for alignment with transformer architecture)  
- **Deployment**: Real-time Chrome Extension using ONNX-optimized BioBERT model via Flask API hosted on Fly.io  

---

## Dataset
1. **Formal Dataset**  
   - Derived from the DETERRENT dataset  
   - 2269 diabetes-related claims (True: 1661, False: 608)  

2. **Informal Dataset**  
   - Curated from Facebook, Twitter (X), and Reddit  
   - Manually annotated into 3 classes (True, False, Partially True)  
   - 902 diabetes-related claims (True: 575, False: 167, Partially True: 160)  
   - Preprocessing included normalization, unicode correction, contraction expansion, emoji/URL filtering  

---

## Model Pipeline
1. **Stage 1**: Supervised fine-tuning of BioBERT on formal two-class data  
2. **Stage 2**: Domain Adaptation using DANN — encoder learns invariant features between formal and informal domains  
3. **Stage 3**: Final supervised fine-tuning on informal three-class data  

---

## Content Filtering (SBERT + Cosine Similarity)
- SBERT model: `all-MiniLM-L6-v2`  
- Averaged embedding vector created from diabetes domain corpus  
- Queries must pass a cosine similarity threshold (> 0.7) to be considered “in-domain”  

---

## Explainability
- **Tried**: SHAP, LIME, Transformers Interpret  
- **Chosen**: Transformers Interpret  
  - Attention-based saliency, integrated gradients  
  - Highlights tokens contributing to prediction  
  - Works seamlessly with subword tokenization  

---

## Deployment

### Backend (Flask API on Fly.io)
- Endpoints:  
  - `/predict` → main prediction route  
  - `/ping` → health check  
- Includes SBERT filtering and explanation generation  
- Deployed via Fly.io with ONNX for faster inference  

**Steps to Deploy on Fly.io (Windows example):**

1. **Install Flyctl**  
   - Download from [Fly.io Download](https://fly.io/docs/hands-on/installing/)  
   - Unzip into `C:\flyctl\`  
   - Run `C:\flyctl\flyctl.exe version` to confirm installation  
   - Authenticate:  
     ```bash
     C:\flyctl\flyctl.exe auth login
     ```

2. **Navigate to project folder**  
   ```bash
   cd "C:\Users\linda\OneDrive\Desktop\DiaBERT_Backend"
