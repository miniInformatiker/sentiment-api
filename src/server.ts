import express, { Request, Response } from 'express';
import path from 'path';

// Rust Addon laden
const addon = require('../rust-addon/sentiment-addon.darwin-arm64.node');

interface SentimentResult {
    label: 'POSITIVE' | 'NEGATIVE';
    score: number;
}

const app = express();
app.use(express.json());

const MODEL_PATH = path.join(__dirname, '../model/model.onnx');
const TOKENIZER_PATH = path.join(__dirname, '../model/tokenizer.json');

app.post('/analyze', (req: Request, res: Response) => {
    const { text } = req.body;

    if (!text) {
        res.status(400).json({ error: 'text fehlt im Request Body' });
        return;
    }

    try {
        const result: SentimentResult = addon.analyze(text, MODEL_PATH, TOKENIZER_PATH);
        res.json(result);
    } catch (err: any) {
        res.status(500).json({ error: err.message });
    }
});

app.listen(3000, () => {
    console.log('🚀 Server läuft auf http://localhost:3000');
});