#!/usr/bin/env python3
"""
Robust SeamlessM4T FLEURS eval script for 8 Indic languages.
Saves per-sample CSVs and per-language JSON summaries (BLEU, chrF++) and an overall summary.

Notes:
- Install: pip install -U transformers datasets sacrebleu tqdm torchaudio
- Run example:
  python sm4t_directmt.py --model facebook/seamless-m4t-v2-large --split test --out_dir results_sm4t --max_samples 1000
"""
import argparse
import json
import numpy as np
from pathlib import Path
import csv
import torch
from datasets import load_dataset, Audio
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from sacrebleu.metrics import BLEU, CHRF
from tqdm import tqdm
import warnings

warnings.filterwarnings("once", category=UserWarning)

# ---------- Correct FLEURS config mapping (use these when loading google/fleurs) ----------
FLEURS_CONFIG = {
    "hi": "hi_in",
    "mr": "mr_in",
    "ta": "ta_in",
    "te": "te_in",
    "kn": "kn_in",
    "bn": "bn_in",
    "gu": "gu_in",
    "ur": "ur_pk",
}

# ---------- SeamlessM4T target short codes (we will try to force the model to these) ----------
SEAMLESS_TGT = {
    "hi": "hin",
    "mr": "mar",
    "ta": "tam",
    "te": "tel",
    "kn": "kan",
    "bn": "ben",
    "gu": "guj",
    "ur": "urd",
}


def load_model_processor(model_name: str, device_choice: str):
    """Load processor and model, return (model, processor, device_str)."""
    device = "cuda" if (device_choice == "auto" and torch.cuda.is_available()) or (device_choice.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name).to(device)
    return model, processor, device


def get_language_forcing_kwargs(processor, seamless_lang: str, device: str):
    """
    Robustly find a method to force the decoder language.
    Returns a dict of kwargs suitable for model.generate (e.g., {"forced_bos_token_id": id} or {"decoder_input_ids": tensor}).
    """
    # 1) tokenizer.get_lang_id(lang)
    tok = getattr(processor, "tokenizer", None)
    if tok is not None:
        get_lang_id = getattr(tok, "get_lang_id", None)
        if callable(get_lang_id):
            try:
                lid = tok.get_lang_id(seamless_lang)
                if lid is not None:
                    return {"forced_bos_token_id": int(lid), "method": "tokenizer.get_lang_id"}
            except Exception:
                pass

        # 2) lang_code_to_id mapping
        lc2id = getattr(tok, "lang_code_to_id", None)
        if lc2id and seamless_lang in lc2id:
            try:
                return {"forced_bos_token_id": int(lc2id[seamless_lang]), "method": "tokenizer.lang_code_to_id"}
            except Exception:
                pass

        # 3) convert token string "<|{lang}|>" -> id
        try:
            token = f"<|{seamless_lang}|>"
            tid = tok.convert_tokens_to_ids(token)
            # some tokenizers return 0 for unk; check unknown token id if exists
            unk = getattr(tok, "unk_token_id", None)
            if tid is not None and tid != unk and tid != 0:
                return {"forced_bos_token_id": int(tid), "method": "tokenizer.convert_tokens_to_ids"}
        except Exception:
            pass

    # 4) processor.get_decoder_prompt_ids(language=..., task="translate")
    get_prompt = getattr(processor, "get_decoder_prompt_ids", None)
    if callable(get_prompt):
        try:
            decoder_prompt = processor.get_decoder_prompt_ids(language=seamless_lang, task="translate")
            if decoder_prompt is not None:
                # convert nested list to tensor of shape (batch, seq)
                import torch as _torch
                t = _torch.tensor(decoder_prompt, dtype=_torch.long).to(device)
                return {"decoder_input_ids": t, "method": "processor.get_decoder_prompt_ids"}
        except Exception:
            pass

    # 5) Last resort: ask generation to accept tgt_lang (some wrappers accept this)
    # Not guaranteed; will be used only if others failed
    try:
        return {"tgt_lang": seamless_lang, "method": "generate.tgt_lang"}
    except Exception:
        pass

    return {"method": None}


def translate_audio(model, processor, audio_array, sr, seamless_lang: str, device: str):
    """
    Translate a single audio numpy array (1D) to target text using robust forcing.
    Returns hypothesis string (may be empty on error).
    """
    # basic checks & normalization
    if audio_array is None:
        raise ValueError("audio_array is None")
    audio_np = np.asarray(audio_array, dtype=np.float32)
    if audio_np.size == 0:
        raise ValueError("Empty audio_array passed")

    # Build processor inputs (use audios=...)
    # Use try/except to handle different processor signatures gracefully.
    try:
        inputs = processor(audios=audio_np, sampling_rate=sr, return_tensors="pt")
    except TypeError:
        # fallback to explicit field name variations
        inputs = processor(audio=audio_np, sampling_rate=sr, return_tensors="pt")
    # move tensors to device
    for k, v in list(inputs.items()):
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)

    # Build gen kwargs robustly
    forcing = get_language_forcing_kwargs(processor, seamless_lang, device)
    method = forcing.pop("method", None)

    if method is None:
        print(f"⚠ Warning: could not find a language-forcing method for '{seamless_lang}'. Model may produce wrong language.")
    else:
        # print small note (not every utt) — but helpful at start of language
        print(f"   -> language-forcing method: {method}")

    # Remove None values in forcing
    gen_kwargs = {k: v for k, v in forcing.items() if v is not None}

    # Ensure max_new_tokens set
    gen_kwargs.setdefault("max_new_tokens", 256)

    # Now try to call generate. Some models require passing input_features directly.
    try:
        generated = model.generate(**inputs, **gen_kwargs)
    except TypeError:
        # try to pass input_features tensor if present
        if "input_features" in inputs:
            try:
                generated = model.generate(inputs["input_features"].to(device), **gen_kwargs)
            except Exception as e:
                # rethrow with context
                raise RuntimeError(f"generate failed with input_features fallback: {e}") from e
        else:
            raise

    # decode safely
    try:
        hyp = processor.batch_decode(generated, skip_special_tokens=True)[0]
    except Exception:
        # last resort: use tokenizer if available
        tok = getattr(processor, "tokenizer", None)
        if tok is not None:
            hyp = tok.decode(generated[0], skip_special_tokens=True)
        else:
            hyp = ""
    return hyp


def load_pairs(fleurs_config_name: str, split: str, sr: int = 16000):
    """
    Load en_us audio and the target-language text (paired by id) from google/fleurs.
    fleurs_config_name: e.g., "hi_in", "bn_in", etc.
    """
    src = load_dataset("google/fleurs", "en_us", split=split)
    tgt = load_dataset("google/fleurs", fleurs_config_name, split=split)
    src = src.cast_column("audio", Audio(sampling_rate=sr))

    ref_map = {}
    for ex in tgt:
        ref_map[int(ex["id"])] = (ex.get("transcription") or ex.get("raw_transcription") or "")

    pairs = []
    for ex in src:
        idx = int(ex["id"])
        if idx in ref_map:
            pairs.append({"id": idx, "audio": ex["audio"]["array"], "sr": ex["audio"]["sampling_rate"], "ref": ref_map[idx]})
    return pairs


def evaluate_and_save(outputs, out_dir: Path, lang_key: str):
    bleu = BLEU(effective_order=True)
    chrf = CHRF(word_order=2)

    csv_path = out_dir / f"per_sample_{lang_key}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "reference", "hypothesis", "BLEU", "chrF++"])
        for o in outputs:
            writer.writerow([o["id"], o["ref"], o["hyp"], f"{o['bleu']:.4f}", f"{o['chrfpp']:.4f}"])

    refs = [o["ref"] for o in outputs]
    hyps = [o["hyp"] for o in outputs]
    corpus_bleu = bleu.corpus_score(hyps, [refs]).score if hyps else 0.0
    corpus_chrf = chrf.corpus_score(hyps, [refs]).score if hyps else 0.0

    summary = {"language": lang_key, "samples": len(outputs), "corpus_bleu": corpus_bleu, "corpus_chrfpp": corpus_chrf}
    with (out_dir / f"summary_{lang_key}.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return corpus_bleu, corpus_chrf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="facebook/seamless-m4t-v2-large")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out_dir", default="results_sm4t")
    parser.add_argument("--max_samples", type=int, default=0)
    args = parser.parse_args()

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model {args.model} ... (this may print HF warnings)")
    model, processor, device = load_model_processor(args.model, args.device)

    overall = {"per_language": {}, "macro_bleu": 0.0, "macro_chrfpp": 0.0}

    for lang_short, fleurs_cfg in FLEURS_CONFIG.items():
        seamless_lang = SEAMLESS_TGT.get(lang_short)
        print(f"\n=== {lang_short} ({fleurs_cfg}) -> Seamless code '{seamless_lang}' ===")

        pairs = load_pairs(fleurs_cfg, args.split)
        if args.max_samples > 0:
            pairs = pairs[: args.max_samples]

        outputs = []
        # print a one-time attempt message about language forcing methods (the function will also print details)
        for ex in tqdm(pairs, desc=f"Translating {lang_short}", unit="utt"):
            try:
                hyp = translate_audio(model, processor, ex["audio"], ex["sr"], seamless_lang, device)
            except Exception as e:
                print(f"  ⚠ error id={ex['id']}: {type(e).__name__}: {e}")
                hyp = ""
            # sentence-level metrics (safe)
            try:
                bleu_sent = BLEU(effective_order=True).sentence_score(hyp, [ex["ref"]]).score if hyp else 0.0
                chrf_sent = CHRF(word_order=2).sentence_score(hyp, [ex["ref"]]).score if hyp else 0.0
            except Exception:
                bleu_sent = 0.0
                chrf_sent = 0.0

            outputs.append({"id": ex["id"], "ref": ex["ref"], "hyp": hyp, "bleu": bleu_sent, "chrfpp": chrf_sent})

        lang_out = outdir / lang_short
        lang_out.mkdir(parents=True, exist_ok=True)
        c_bleu, c_chrf = evaluate_and_save(outputs, lang_out, lang_short)
        overall["per_language"][lang_short] = {"corpus_bleu": c_bleu, "corpus_chrfpp": c_chrf, "samples": len(outputs)}

    # macro averages
    n = len(overall["per_language"])
    if n:
        overall["macro_bleu"] = sum(v["corpus_bleu"] for v in overall["per_language"].values()) / n
        overall["macro_chrfpp"] = sum(v["corpus_chrfpp"] for v in overall["per_language"].values()) / n

    with (outdir / "overall_summary.json").open("w", encoding="utf-8") as f:
        json.dump(overall, f, ensure_ascii=False, indent=2)

    print("\n=== DONE ===")
    print(json.dumps(overall, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
