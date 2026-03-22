# Sentiment API

Sentiment-Analyse API mit Node.js + TypeScript + Rust (ONNX)

## Setup

### 1. Modell herunterladen

\```bash
uv venv
source .venv/bin/activate
uv add "optimum[onnxruntime]" "transformers>=4.36.0,<4.58.0"
uv run export_model.py
\```

### 2. Rust Addon bauen

\```bash
cd rust-addon
cargo build
npm run build
cd ..
\```

### 3. Server starten

\```bash
npm install
npx ts-node src/server.ts
\```

## Verwendung

\```bash
curl -X POST http://localhost:3000/analyze \
 -H "Content-Type: application/json" \
 -d '{"text": "This movie was absolutely amazing!"}'
\```
