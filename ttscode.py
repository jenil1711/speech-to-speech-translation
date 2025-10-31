import os
import torch
import torchaudio
from TTS.api import TTS
import jiwer
import numpy as np
import parselmouth
from scipy.spatial.distance import euclidean
import csv

# ---------------------------
#  CONFIG
# ---------------------------
PARENT_DIR = "/home/jenil/SM4T_MT/results_sm4t"
OUTPUT_DIR = "/home/jenil/SM4T_MT/SM4T_TTS"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Vakyansh TTS model (change per language if needed)
# Example: Hindi model
TTS_MODEL = "vakyansh-tts-hindi"  
tts = TTS(TTS_MODEL).to(device)

# ---------------------------
#  EVALUATION HELPERS
# ---------------------------

def compute_mcd(ref_mel, gen_mel):
    """Mel Cepstral Distortion (requires same length)"""
    min_len = min(ref_mel.shape[0], gen_mel.shape[0])
    ref, gen = ref_mel[:min_len], gen_mel[:min_len]
    dist = np.mean([euclidean(r, g) for r, g in zip(ref, gen)])
    return dist

def compute_f0_rmse(ref_audio, gen_audio, sr=22050):
    ref_pitch = parselmouth.Sound(ref_audio, sr).to_pitch().selected_array["frequency"]
    gen_pitch = parselmouth.Sound(gen_audio, sr).to_pitch().selected_array["frequency"]
    min_len = min(len(ref_pitch), len(gen_pitch))
    return np.sqrt(np.mean((ref_pitch[:min_len] - gen_pitch[:min_len])**2))

def compute_wer(ref_text, gen_text):
    return jiwer.wer(ref_text, gen_text)

def compute_mos_placeholder(mcd, f0_rmse, wer):
    """Pseudo-MOS (formulaic, not human-rated)"""
    mos = 5 - (mcd/10 + f0_rmse/50 + wer*2)
    return max(1.0, min(5.0, mos))

# ---------------------------
#  MAIN LOOP
# ---------------------------
results_file = os.path.join(OUTPUT_DIR, "tts_evaluations.csv")

with open(results_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Language", "File", "WER", "MCD", "F0_RMSE", "MOS"])

    for lang in os.listdir(PARENT_DIR):
        lang_dir = os.path.join(PARENT_DIR, lang)
        if not os.path.isdir(lang_dir):
            continue

        print(f"Processing {lang}...")

        lang_out_dir = os.path.join(OUTPUT_DIR, lang)
        os.makedirs(lang_out_dir, exist_ok=True)

        for file in os.listdir(lang_dir):
            if not file.endswith(".txt"):
                continue

            file_path = os.path.join(lang_dir, file)

            # Read MT output
            with open(file_path, "r", encoding="utf-8") as txt_file:
                text = txt_file.read().strip()

            # Generate TTS
            gen_audio_path = os.path.join(lang_out_dir, file.replace(".txt", ".wav"))
            tts.tts_to_file(text=text, file_path=gen_audio_path)

            # Dummy reference: use same text (if GT speech is available, replace this step)
            ref_audio = gen_audio_path  
            gen_audio = gen_audio_path

            # Load audio
            ref_waveform, sr = torchaudio.load(ref_audio)
            gen_waveform, _ = torchaudio.load(gen_audio)

            # Compute features
            ref_mel = torchaudio.transforms.MelSpectrogram()(ref_waveform)
            gen_mel = torchaudio.transforms.MelSpectrogram()(gen_waveform)

            mcd = compute_mcd(ref_mel.squeeze().T.numpy(), gen_mel.squeeze().T.numpy())
            f0_rmse = compute_f0_rmse(ref_waveform.squeeze().numpy(), gen_waveform.squeeze().numpy(), sr)
            wer = compute_wer(text, text)  # placeholder: needs ground truth transcript
            mos = compute_mos_placeholder(mcd, f0_rmse, wer)

            # Save results
            writer.writerow([lang, file, wer, mcd, f0_rmse, mos])

print(f"✅ Completed. Results stored at {results_file}")
