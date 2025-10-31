import os
import torch
import pandas as pd
import json
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from datasets import load_dataset, Audio
from evaluate import load
from tqdm import tqdm
import unicodedata
import re

# Config
MODEL_NAME = "facebook/hf-seamless-m4t-large"
DEVICE = 0 if torch.cuda.is_available() else -1
BATCH_SIZE = 8
LANG_CODES = ["hi_in", "gu_in", "ta_in", "te_in", "bn_in", "mr_in", "ur_pk", "kn_in"]
MAX_SAMPLES = 1000
RESULTS_BASE = "indic-eng-results-s2t"
 
os.makedirs(RESULTS_BASE, exist_ok=True)

# Normalization
def normalize(text):
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip().lower()

print("[INFO] Loading SeamlessM4T model and processor...")
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_NAME).to(DEVICE).eval()

def run_batch_infer(audio_arrays, tgt_lang="eng"):
    inputs = processor(audios=audio_arrays, return_tensors="pt", sampling_rate=16000)
    input_features = inputs["input_features"].to(DEVICE)
    with torch.no_grad():
        generated_ids = model.generate(input_features=input_features, tgt_lang=tgt_lang)
    outputs = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return outputs

wer_metric = load("wer")
cer_metric = load("cer")

# --- Load English references (en_us) once ---
print("[INFO] Loading English references (en_us split)...")
eng_ds = load_dataset("google/fleurs", name="en_us", split=f"test[:{MAX_SAMPLES}]")
id2eng = {ex["id"]: ex["transcription"] for ex in eng_ds}

for lang in LANG_CODES:
    print(f"\n[INFO] Processing: {lang}")
    ds = load_dataset("google/fleurs", name=lang, split=f"test[:{MAX_SAMPLES}]")
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    preds, refs, meta = [], [], []
    audio_batch, ex_ids, ref_batch = [], [], []

    for i, ex in enumerate(tqdm(ds, desc=f"Infer({lang})")):
        audio_array = ex["audio"]["array"]
        utt_id = ex["id"]
        ref_english = id2eng.get(utt_id, None)
        if audio_array is None or ref_english is None:
            continue
        audio_batch.append(audio_array)
        ex_ids.append(utt_id)
        ref_batch.append(ref_english)
        batch_full = (len(audio_batch) == BATCH_SIZE) or (i == len(ds)-1)
        if batch_full:
            out_texts = run_batch_infer(audio_batch, tgt_lang="eng")
            preds.extend([normalize(o) for o in out_texts])
            refs.extend([normalize(r) for r in ref_batch])
            meta.extend(list(zip(ex_ids, out_texts, ref_batch)))
            audio_batch, ex_ids, ref_batch = [], [], []

    # Save results
    out_dir = os.path.join(RESULTS_BASE, lang)
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(meta, columns=["id", "prediction", "reference"])
    df.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)
    with open(os.path.join(out_dir, "predictions.jsonl"), "w", encoding="utf-8") as f:
        for row in meta:
            json.dump({"id": row[0], "prediction": row[1], "reference": row[2]}, f, ensure_ascii=False)
            f.write("\n")

    # Evaluation
    if preds and refs:
        wer = wer_metric.compute(predictions=preds, references=refs)
        cer = cer_metric.compute(predictions=preds, references=refs)
        print(f"[RESULTS] {lang}: WER={wer:.3f} CER={cer:.3f} [Samples: {len(preds)}]")
        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump({"WER": wer, "CER": cer, "num_samples": len(preds)}, f, indent=2)
    else:
        print(f"[WARN] No valid predictions or references for language {lang}")

print("\n✅ All done!")
