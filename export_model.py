from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
OUTPUT_DIR = "./model"

print("⏳ Lade Modell von Hugging Face herunter...")

model = ORTModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    export=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"✅ Modell gespeichert in: {OUTPUT_DIR}/")