import json
import os
import librosa
import parselmouth as pm
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, Audio
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
from tqdm import tqdm
import soundfile as sf
from nisqa.nisqa_model import nisqaModel
import sacrebleu
import sys, os
sys.path.append(os.path.dirname(__file__))

LANG_CODES = ["hi_in","gu_in","ta_in","te_in","bn_in","mr_in","ur_pk","kn_in"]
MT_RESULTS_BASE = "/home/jenil/Codefiles/seamlessm4t"
TTS_OUT_DIR = "I2E_tts_audios"
EVAL_OUT_DIR = "tts_eval_results"
os.makedirs(TTS_OUT_DIR, exist_ok=True)
os.makedirs(EVAL_OUT_DIR, exist_ok=True)

# ---- TTS setup: Parler-TTS ----
TTS_MODEL = "parler-tts/parler-tts-mini-v1"   # try "parler-tts/parler-tts-large-v1" for better quality
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = ParlerTTSForConditionalGeneration.from_pretrained(TTS_MODEL).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(TTS_MODEL)
desc_tokenizer = AutoTokenizer.from_pretrained(TTS_MODEL)
text_tokenizer = desc_tokenizer
SR = model.config.sampling_rate if hasattr(model.config, "sampling_rate") else 16000

# ---- NISQA setup (MOS) ----
args = {
    "device": "cuda",
    "mode": "predict_file",
    "deg": "/home/jenil/Codefiles/NISQA/results/degraded/example_audio.wav",
    "csv_eval": "/home/jenil/Codefiles/NISQA/results/eval/example_output.csv",
    "pretrained_model": "/home/jenil/Codefiles/NISQA/weights/nisqa.tar",
    "ms_channel": "mono"
}
nisqa = nisqaModel(args)
nisqa.preprocess_batch = 32

# ---- Speaker Feature Extractor (simplified: only pitch) ----
class VoiceCharacteristicsExtractor:
    def __init__(self, sr=16000):
        self.sr = sr

    def extract(self, audio):
        features = {}
        snd = pm.Sound(audio, self.sr)
        pitch_obj = snd.to_pitch()
        f0_vals = pitch_obj.selected_array['frequency']
        voiced = f0_vals[(f0_vals > 50) & (f0_vals < 500)]
        features["mean_pitch_hz"] = float(np.mean(voiced)) if voiced.size > 0 else 0.0
        return features, self.describe(features)

    def describe(self, feats):
        # Pitch-based gender mapping
        if feats["mean_pitch_hz"] > 180:
            pitch = "a female voice"
        elif feats["mean_pitch_hz"] < 120 and feats["mean_pitch_hz"] > 0:
            pitch = "a male voice"
        else:
            pitch = "a neutral voice"
        return f"A clear {pitch} speaking in English."

extractor = VoiceCharacteristicsExtractor()

# ---- Batched TTS ----
def batch_generate_tts(model, desc_tokenizer, text_tokenizer, device, texts, descriptions, out_paths):
    desc_inputs = desc_tokenizer(list(descriptions), return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
    text_inputs = text_tokenizer(list(texts), return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
    with torch.no_grad():
        wavs = model.generate(
            input_ids=desc_inputs,
            prompt_input_ids=text_inputs,
            do_sample=False,           # deterministic output
            max_length=1024
        )
    for w, path in zip(wavs, out_paths):
        audio_arr = w.cpu().numpy().squeeze()
        # normalize before saving
        audio_arr = audio_arr / (np.max(np.abs(audio_arr)) + 1e-6)
        sf.write(path, audio_arr, SR)

# ---- Evaluation metrics ----
def compute_mcd(ref, syn, sr=16000):
    ref = librosa.feature.mfcc(y=ref, sr=sr)
    syn = librosa.feature.mfcc(y=syn, sr=sr)
    min_len = min(ref.shape[1], syn.shape[1])
    ref, syn = ref[:,:min_len], syn[:,:min_len]
    diff = ref - syn
    mcd = np.mean(np.sqrt(np.sum(diff**2, axis=0)))
    return float(mcd)

def compute_f0_rmse(ref, syn, sr=16000):
    f0_ref = librosa.yin(ref, fmin=50, fmax=500, sr=sr)
    f0_syn = librosa.yin(syn, fmin=50, fmax=500, sr=sr)
    min_len = min(len(f0_ref), len(f0_syn))
    err = np.sqrt(np.mean((f0_ref[:min_len] - f0_syn[:min_len])**2))
    return float(err)

def compute_cer(hyp, ref):
    import editdistance
    return editdistance.eval(hyp, ref) / max(1, len(ref))

# ---- Main loop per language ----
for lang in LANG_CODES:
    print(f"\n=== {lang} ===")
    mt_csv = os.path.join(MT_RESULTS_BASE, lang, "predictions.csv")
    if not os.path.exists(mt_csv):
        print(f"Missing MT results for {lang}")
        continue
    mt_df = pd.read_csv(mt_csv)
    ds = load_dataset("google/fleurs", lang, split=f"test[:{len(mt_df)}]")
    ds = ds.cast_column("audio", Audio(sampling_rate=extractor.sr))
    id2audio = {row["id"]: row["audio"]["array"] for row in ds}

    utt_ids, eng_texts, features, descriptions, gen_wav_paths = [], [], [], [], []
    for _, row in mt_df.iterrows():
        utt_id = row["id"]
        eng_text = str(row["prediction"])
        if utt_id not in id2audio or not eng_text.strip():
            continue
        audio = id2audio[utt_id]
        feats, desc = extractor.extract(audio)
        utt_ids.append(utt_id)
        eng_texts.append(eng_text)
        features.append(feats)
        descriptions.append(desc)
        out_path = os.path.join(TTS_OUT_DIR, lang, f"{utt_id}.wav")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        gen_wav_paths.append(out_path)

    batch_size = 16
    for i in range(0, len(utt_ids), batch_size):
        batch_texts = eng_texts[i:i+batch_size]
        batch_descs = descriptions[i:i+batch_size]
        batch_paths = gen_wav_paths[i:i+batch_size]
        batch_generate_tts(model, desc_tokenizer, text_tokenizer, DEVICE, batch_texts, batch_descs, batch_paths)

    wav_files = gen_wav_paths
    mos_preds = nisqa.predict_files(wav_files, save_scores=None, eval_mode="mos")
    mos_scores = mos_preds["mos_pred"].tolist()

    fleurs_en = load_dataset("google/fleurs", name="en_us", split=f"test[:{len(utt_ids)}]")
    id2eng = {row["id"]: row["transcription"] for row in fleurs_en}
    refs_text = [id2eng.get(uid, "") for uid in utt_ids]
    refs_wav = []
    id2wav_en = {row["id"]: row["audio"]["array"] for row in fleurs_en}
    for uid in utt_ids:
        wav_en = id2wav_en.get(uid, None)
        ref_wavf = f"/tmp/ref_{uid}.wav"
        sf.write(ref_wavf, wav_en, SR)
        refs_wav.append(wav_en)

    mcds, f0_rmses, cers = [], [], []
    for synth_path, ref_audio, ref_text, hyp_text in tqdm(zip(gen_wav_paths, refs_wav, refs_text, eng_texts), total=len(utt_ids), desc="Eval"):
        try:
            synth_audio, _ = librosa.load(synth_path, sr=SR)
            mcds.append(compute_mcd(ref_audio, synth_audio))
            f0_rmses.append(compute_f0_rmse(ref_audio, synth_audio))
            cers.append(compute_cer(hyp_text, ref_text))
        except Exception:
            mcds.append(np.nan), f0_rmses.append(np.nan), cers.append(np.nan)

    out_eval_df = pd.DataFrame({
        "id": utt_ids,
        "eng_text": eng_texts,
        "ref_text": refs_text,
        "mos": mos_scores,
        "mcd": mcds,
        "f0_rmse": f0_rmses,
        "cer": cers,
        "desc": descriptions
    })
    out_eval_df.to_csv(os.path.join(EVAL_OUT_DIR, f"{lang}_tts_eval.csv"), index=False)
    summary = {
        "avg_mos": np.nanmean(mos_scores),
        "avg_mcd": np.nanmean(mcds),
        "avg_f0_rmse": np.nanmean(f0_rmses),
        "avg_cer": np.nanmean(cers),
        "num_samples": len(mos_scores),
    }
    with open(os.path.join(EVAL_OUT_DIR, f"{lang}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"{lang}: MOS={summary['avg_mos']:.2f} | MCD={summary['avg_mcd']:.2f} | F0 RMSE={summary['avg_f0_rmse']:.2f} | CER={summary['avg_cer']:.2f}")

print("\n✅ Batched Indic-to-English TTS generation + full evaluation complete!")

