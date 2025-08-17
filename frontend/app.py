from flask import Flask, request, jsonify
import torch
import openai
import os
import numpy as np
import onnxruntime as ort
import torch.nn as nn
from transformers import AutoModel
from transformers import AutoConfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers_interpret import SequenceClassificationExplainer
from sentence_transformers import SentenceTransformer, util
from transformers.modeling_outputs import SequenceClassifierOutput
from flask_cors import CORS

app = Flask(__name__)

# Global variables
tokenizer = None
onnx_model = None
pytorch_model = None
explainer = None
sbert_model = None
dataset_embeddings = None
dataset_texts = None
client = None
resources_loaded = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
label_mapping = {0: "real", 1: "false", 2: "partially_true"}

# Load OpenAI key ONCE at the top
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
    client = openai
    print("OpenAI Key Found:", OPENAI_API_KEY[:6], "...")
else:
    print("OPENAI_API_KEY not found in environment.")
    client = None

# Load tokenizer, ONNX, SBERT, etc.
def load_resources():
    global tokenizer, onnx_model, sbert_model, dataset_embeddings, dataset_texts, resources_loaded

    if resources_loaded:
        return

    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
    onnx_model = ort.InferenceSession("./newbiobert_finetuned_3class.onnx")
    sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    dataset_embeddings = torch.load("./combined_embeddings.pt", map_location="cpu")
    dataset_texts = torch.load("./combined_texts.pt", map_location="cpu")

    resources_loaded = True

@app.before_request
def ensure_resources_loaded():
    if not resources_loaded:
        load_resources()

CORS(app, origins=["chrome-extension://pkbfgfhddafhlnndmcnhahnokddbgjjo"],
     allow_headers=["Content-Type"], supports_credentials=True, methods=["GET", "POST", "OPTIONS"])

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Hybrid ONNX + Explainable BioBERT API is running!"})

#ONNX classification
def classify_with_onnx(text):
    encoded = tokenizer(text, return_tensors="np", truncation=True, padding="max_length", max_length=128)
    inputs_onnx = {"input_ids": encoded["input_ids"].astype(np.int64),"attention_mask": encoded["attention_mask"].astype(np.int64)}


    logits = onnx_model.run(["logits"], inputs_onnx)[0]
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    pred_index = int(np.argmax(probs))
    label = label_mapping[pred_index]
    return label, {
        "real": round(probs[0][0]*100, 2),
        "false": round(probs[0][1]*100, 2),
        "partially_true": round(probs[0][2]*100, 2)
    }

#PyTorch fine-tuned model
class FineTuneModel(nn.Module):
    def __init__(self, num_classes=3, hidden_size=768):
        super(FineTuneModel, self).__init__()
        self.encoder = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask=None, labels=None):
        if attention_mask is None:
            attention_mask = (input_ids != tokenizer.pad_token_id).long()
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = output.last_hidden_state[:, 0, :]
        logits = self.classifier(pooled)
        return SequenceClassifierOutput(logits=logits)

    def get_input_embeddings(self):
        return self.encoder.get_input_embeddings()

#Load explainability
def get_explainer():
    global pytorch_model, explainer
    if explainer is None:
        pytorch_model = FineTuneModel(num_classes=3)
        pytorch_model.load_state_dict(
            #torch.load("./newbiobert_model_3class/newbiobert_model_3class/pytorch_model.bin", map_location="cpu")
            torch.load("./newbiobert_model_3class/newbiobert_model_3class/pytorch_model.bin", map_location=torch.device("cpu"))

        )
        pytorch_model.eval()
        config = AutoConfig.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
        pytorch_model.config = config
        pytorch_model.base_model_prefix = "bert"
        pytorch_model.device = torch.device("cpu")
        pytorch_model.bert = pytorch_model.encoder
        explainer = SequenceClassificationExplainer(pytorch_model, tokenizer)
    return explainer


#Explanation using GPT with merged subwords
def get_explanation(text, label, confidence):
    explainer = get_explainer()
    attributions = explainer(text)

    # Manually merge subword attributions
    merged_tokens = []
    merged_scores = []
    current_token = ""
    current_score = 0.0

    for token, score in attributions:
        if token.startswith("##"):
            current_token += token[2:]
            current_score += score
        else:
            if current_token:
                merged_tokens.append(current_token)
                merged_scores.append(current_score)
            current_token = token
            current_score = score

    if current_token:
        merged_tokens.append(current_token)
        merged_scores.append(current_score)

    token_score_pairs = list(zip(merged_tokens, merged_scores))
    token_score_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    keywords = [w for w, s in token_score_pairs[:7] if abs(s) > 0.05]  # still filter for meaningfully contributing tokens
   



    label_map = {
        "real": "True Information",
        "false": "False Information",
        "partially_true": "Partially True / Misleading Information"
    }
    mapped_label = label_map.get(label, label)

    if not client:
        return "OpenAI key not set.", keywords
    



    prompt = f"""
    The model classified the following text as '{mapped_label}' with {confidence:.1f}% confidence.

    Text: "{text}"

    The model highlighted these key words: {', '.join(keywords)}.
    
    Briefly explain why the model likely made this prediction, using only the most meaningful highlighted words from the model. 
    You may ignore trivial or stopwords. 
    Make your explanation faithful to the meaning of the sentence and how the words influence the label.

    Then, write a short follow-up sentence that adds helpful context, clarification, or health guidance related to the topic and Offers medically relevant insight based on the input.

    """

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=150
        )
        return response.choices[0].message.content.strip(), keywords
    except Exception as e:
        print("GPT Error:", e)
        return "Explanation failed.", keywords

#SBERT similarity check
def is_query_relevant(text, threshold=0.69):
    embedding = sbert_model.encode(text, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(embedding, dataset_embeddings)
    return torch.max(scores).item() >= threshold


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "No text provided."}), 400

        if not text.endswith((".", "!", "?")):
            text += "."
        text = text[0].upper() + text[1:]

        if not is_query_relevant(text):
            return jsonify({
                "is_relevant": False,
                "message": "Query not related to diabetes."
            })

        label, raw_probs = classify_with_onnx(text)
        probs = {
            "real": float(raw_probs["real"]),
            "false": float(raw_probs["false"]),
            "partially_true": float(raw_probs["partially_true"])
        }

        explanation, keywords = get_explanation(text, label, probs[label])

        return jsonify({
            "text": text,
            "is_relevant": True,
            "predicted_label": label,
            "probabilities": probs,
            "key_words": keywords,
            "explanation": explanation
        })
    except Exception as e:
        import traceback
        print("ERROR in /predict:", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/ping")
def ping():
    return "DiaBERT server is up!", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
