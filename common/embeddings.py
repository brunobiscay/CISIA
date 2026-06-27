import re

import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "almanach/camembert-bio-base"  # modèle FR médical

_tokenizer = None
_model = None


def load_model():
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModel.from_pretrained(MODEL_NAME)
        _model.eval()
    return _tokenizer, _model


def clean_text(text) -> str:
    if pd.isna(text):
        return "aucun symptome"

    text = str(text)
    text = re.sub(r"\d+", "", text)

    return text.strip()


def compute_embedding(text: str) -> list[float]:
    tokenizer, model = load_model()

    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    # mean pooling
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze()

    return embedding.tolist()
