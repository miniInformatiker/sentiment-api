# Sentiment API – CLAUDE.md

## Projektbeschreibung

Sentiment-Analyse API: Node.js/TypeScript Server mit Rust Addon (napi-rs)
das ein vortrainiertes DistilBERT ONNX Modell ausführt.

## Stack

- **Node.js + TypeScript** – HTTP Server (Express)
- **Rust (napi-rs)** – Natives Addon für ML-Inferenz
- **ort 2.0.0-rc.12** – ONNX Runtime Rust Binding
- **tokenizers 0.21** – Hugging Face Tokenizer in Rust
- **Modell** – distilbert-base-uncased-finetuned-sst-2-english (SST-2, 2 Klassen)

## Projektstruktur

```
sentiment-api/
├── src/server.ts         # Express Server
├── rust-addon/src/lib.rs # Rust Addon (ML-Inferenz)
├── model/                # ONNX Modell (nicht im Git)
├── export_model.py       # Modell von HuggingFace herunterladen
└── CLAUDE.md
```

## Aktueller Stand

- ✅ Baustein 1: ONNX Modell exportiert
- ✅ Baustein 2: Rust Addon mit napi-rs aufgesetzt
- ✅ Baustein 3: Rust Code (Session, Tokenizer, Inferenz)
- ✅ Baustein 4: TypeScript Server läuft
- ⏳ Offen: Batch-Verarbeitung (mehrere Texte auf einmal)
- ⏳ Offen: Neutral-Klasse (3-Klassen Modell oder Score-Schwellenwert)

## Bekannte Eigenheiten

- Modell kennt nur POSITIVE/NEGATIVE, keine NEUTRAL Klasse
- .node Datei heißt sentiment-addon.darwin-arm64.node (Mac M-Chip)
- ndarray muss Version 0.17 sein (ort Kompatibilität)
- transformers muss <4.58.0 sein (optimum Kompatibilität)

## Nächste Schritte

1. Batch-Verarbeitung implementieren
2. Optional: Auf 3-Klassen Modell wechseln
