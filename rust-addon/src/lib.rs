use napi_derive::napi;
use ndarray::Array2;
use ort::session::Session;
use ort::value::Tensor;
use tokenizers::Tokenizer;

#[napi(object)]
pub struct SentimentResult {
  pub label: String,
  pub score: f64,
}

#[napi]
pub fn analyze(
  text: String,
  model_path: String,
  tokenizer_path: String,
) -> napi::Result<SentimentResult> {
  // Session laden
  let mut session = Session::builder()
    .map_err(|e| napi::Error::from_reason(e.to_string()))?
    .commit_from_file(&model_path)
    .map_err(|e| napi::Error::from_reason(e.to_string()))?;

  // Tokenizer laden
  let tokenizer =
    Tokenizer::from_file(&tokenizer_path).map_err(|e| napi::Error::from_reason(e.to_string()))?;

  // Text tokenisieren
  let encoding = tokenizer
    .encode(text.as_str(), true)
    .map_err(|e| napi::Error::from_reason(e.to_string()))?;

  let ids: Vec<i64> = encoding.get_ids().iter().map(|x| *x as i64).collect();
  let mask: Vec<i64> = encoding
    .get_attention_mask()
    .iter()
    .map(|x| *x as i64)
    .collect();
  let len = ids.len();

  // 2D Arrays für das Modell [batch=1, sequence_length]
  let ids_array =
    Array2::from_shape_vec((1, len), ids).map_err(|e| napi::Error::from_reason(e.to_string()))?;
  let mask_array =
    Array2::from_shape_vec((1, len), mask).map_err(|e| napi::Error::from_reason(e.to_string()))?;

  // Tensoren erstellen
  let ids_tensor =
    Tensor::from_array(ids_array).map_err(|e| napi::Error::from_reason(e.to_string()))?;
  let mask_tensor =
    Tensor::from_array(mask_array).map_err(|e| napi::Error::from_reason(e.to_string()))?;

  // Modell ausführen
  let outputs = session
    .run(ort::inputs![ids_tensor, mask_tensor])
    .map_err(|e| napi::Error::from_reason(e.to_string()))?;

  // Logits auslesen
  let logits = outputs[0]
    .try_extract_array::<f32>()
    .map_err(|e| napi::Error::from_reason(e.to_string()))?;
  let logits: Vec<f32> = logits.iter().cloned().collect();

  // Softmax → Wahrscheinlichkeiten
  let exp: Vec<f32> = logits.iter().map(|x| x.exp()).collect();
  let sum: f32 = exp.iter().sum();
  let probs: Vec<f32> = exp.iter().map(|x| x / sum).collect();

  // 0 = NEGATIVE, 1 = POSITIVE
  let (label, score) = if probs[1] > probs[0] {
    ("POSITIVE", probs[1])
  } else {
    ("NEGATIVE", probs[0])
  };

  Ok(SentimentResult {
    label: label.to_string(),
    score: score as f64,
  })
}
