# ef_training_final_plateau_safe_resume.py
import os, json, random, warnings, gc
from glob import glob
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from transformers import (
    SeamlessM4TProcessor,
    SeamlessM4TForSpeechToText,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
import sacrebleu

# ----------------------------
# Safe allowlist for PyTorch 2.6+ (numpy deserialization)
# ----------------------------
try:
    torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
except Exception as e:
    print("Safe global add:", e)

# ----------------------------
# CONFIG
# ----------------------------
OLD_TRAIN = "/home/jenil/seamless_communication/large_quality_data/train_dataset.pt"
OLD_VAL = "/home/jenil/seamless_communication/large_quality_data/val_dataset.pt"

NEW_TRAIN_DIR = "./part-2_dataset"
PART2_VAL = "/home/jenil/seamless_communication/large_quality_data/val_merged.pt"
OUTPUT_DIR = "./3rd_finetune_final"

TARGET_LANG = "eng"
SEED = 42
SAMPLE_RATE = 16000
CHUNK_SECS = 8
OVERLAP_SECS = 1
CHUNK_LEN = CHUNK_SECS * SAMPLE_RATE
OVERLAP = OVERLAP_SECS * SAMPLE_RATE

PER_DEVICE_TRAIN_BS = 1
GRADIENT_ACCUMULATION = 6
PER_DEVICE_EVAL_BS = 1

LEARNING_RATE = 4e-4
MIN_LR = 1e-6
EPOCHS = 5
WEIGHT_DECAY = 0.01

LORA_R = 48
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

MAX_TARGET_LEN = 128
USE_FP16 = True
LOGGING_STEPS = 100
REPORT_TO = None
SAVE_STEPS = 2000
SAVE_TOTAL_LIMIT = 5
EARLY_STOP_PATIENCE = 4
WARMUP_RATIO = 0.05

PLATEAU_PATIENCE = 4
PLATEAU_FACTOR = 0.7
PLATEAU_MIN_LR = MIN_LR

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# Repro
# ----------------------------
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
warnings.filterwarnings("ignore", message=".*Token indices sequence length is longer.*")
os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"

if torch.cuda.is_available():
    try:
        torch.cuda.set_per_process_memory_fraction(25.0 / 80.0, 0)
    except Exception:
        pass

# ----------------------------
# Lazy dataset
# ----------------------------
class LazyChunkedPTDataset(Dataset):
    def __init__(self, pt_files, processor, chunk_len=CHUNK_LEN, overlap=OVERLAP, sr=SAMPLE_RATE, max_target_len=MAX_TARGET_LEN):
        self.processor = processor
        self.chunk_len = int(chunk_len)
        self.overlap = int(overlap)
        self.sr = sr
        self.max_target_len = max_target_len
        self.samples = []
        self.segments = []

        print("🔎 Indexing PT files (lazy) ...")
        for f in tqdm(pt_files, desc="Files"):
            if os.path.exists(f):
                try:
                    data = torch.load(f, map_location="cpu", weights_only=False)
                except Exception as e:
                    print(f"⚠️ Failed to load {f}: {e}")
                    continue
                if isinstance(data, (list, tuple)):
                    self.samples.extend(data)
                elif isinstance(data, dict):
                    self.samples.append(data)
                else:
                    try:
                        for el in data:
                            self.samples.append(el)
                    except Exception:
                        print(f"⚠️ Unrecognized format in {f}, skipping.")
                        continue

        for s_idx, s in enumerate(tqdm(self.samples, desc="Indexing samples", unit="sample")):
            audio = None
            for key in ["audio_array", "audio", "chunked_audio_array"]:
              if key in s and s[key] is not None:
                audio = s[key]
                break
            if audio is None:
                raise ValueError(f"No valid audio found in sample {s_idx} (sample keys: {list(s.keys())})")
            if isinstance(audio, torch.Tensor):
                audio = audio.cpu().numpy()
            length = len(audio)
            if length <= self.chunk_len:
                self.segments.append((s_idx, 0, length))
            else:
                start = 0
                while start < length:
                    end = min(length, start + self.chunk_len)
                    self.segments.append((s_idx, start, end))
                    if end == length:
                        break
                    start = max(0, end - self.overlap)
        print(f"📦 Indexed {len(self.samples)} samples -> {len(self.segments)} segments")

    def __len__(self):
        return len(self.segments)

    # ----------------------------
    # Safe __getitem__
    # ----------------------------
    def __getitem__(self, idx):
        try:
             s_idx, start, end = self.segments[idx]
             s = self.samples[s_idx]
             audio = None
             for key in ["audio_array", "audio", "chunked_audio_array"]:
                if key in s and s[key] is not None:
                    audio = s[key]
                    break
             if audio is None:
                raise ValueError(f"No valid audio in sample {s_idx}")
             if isinstance(audio, torch.Tensor):
                audio = audio.cpu().numpy()
             segment = audio[start:end]
             sampling_rate = s.get("sampling_rate", self.sr)
             text = (
                 s.get("target_text") or
                 s.get("text") or
                 s.get("translation_text") or
                 s.get("source_text") or
                 s.get("en_text") or
                 s.get("pred_text") or
                 ""
            )
             inputs = self.processor(
                 audios=segment,
                 sampling_rate=sampling_rate,
                 return_tensors="pt",
                 padding=False,
                 truncation=True,
                 max_length=self.chunk_len,
            )    
             labels = self.processor(
                 text=text,
                 return_tensors="pt",
                 padding=False,
                 truncation=True,
                 max_length=self.max_target_len,
            ).input_ids
             input_features = inputs.input_features.squeeze(0)
             attention_mask = getattr(inputs, "attention_mask", None)
             if attention_mask is not None:
                 attention_mask = attention_mask.squeeze(0)
             labels = labels.squeeze(0)
             return {"input_features": input_features, "labels": labels, "sample_idx": s_idx, "attention_mask": attention_mask}
        except Exception as e:
             print(f"⚠️ Skipping sample {idx} due to: {e}")
             return None


# ----------------------------
# Collate function (safe)
# ----------------------------
def collate_fn(batch):
    # Remove any None entries
    batch = [b for b in batch if b is not None]
    if not batch:
        return {
            "input_features": torch.zeros(1, 100, 80),
            "labels": torch.zeros(1, dtype=torch.long),
            "sample_idx": torch.zeros(1, dtype=torch.long),
        }

    input_feats = nn.utils.rnn.pad_sequence(
        [b["input_features"] for b in batch], batch_first=True, padding_value=0.0
    )
    labels = nn.utils.rnn.pad_sequence(
        [b["labels"] for b in batch], batch_first=True, padding_value=-100
    )

    # Safely handle missing sample_idx
    if all("sample_idx" in b for b in batch):
        sample_idx = torch.tensor([b["sample_idx"] for b in batch], dtype=torch.long)
    else:
        sample_idx = torch.zeros(len(batch), dtype=torch.long)

    out = {"input_features": input_feats, "labels": labels, "sample_idx": sample_idx}

    # Optional attention_mask
    if any("attention_mask" in b and b["attention_mask"] is not None for b in batch):
        masks = nn.utils.rnn.pad_sequence(
            [b.get("attention_mask", torch.zeros(b["input_features"].size(0))) for b in batch],
            batch_first=True, padding_value=0
        )
        out["attention_mask"] = masks

    return out



# ----------------------------
# Metrics
# ----------------------------
def compute_metrics(preds, refs):
    bleu = sacrebleu.corpus_bleu(preds, [refs])
    chrf = sacrebleu.corpus_chrf(preds, [refs])
    return {"bleu": bleu.score, "chrf++": chrf.score}

def merge_segment_preds_to_full(pred_segs, ref_segs, sample_idxs):
    merged_preds = defaultdict(list)
    merged_refs = defaultdict(list)
    for p, r, s in zip(pred_segs, ref_segs, sample_idxs):
        merged_preds[int(s)].append(p)
        merged_refs[int(s)].append(r)
    full_preds = [" ".join(merged_preds[k]) for k in sorted(merged_preds.keys())]
    full_refs = [" ".join(merged_refs[k]) for k in sorted(merged_refs.keys())]
    return full_preds, full_refs


# ----------------------------
# Safe Trainer & LR Plateau
# ----------------------------
class SafeSeq2SeqTrainer(Seq2SeqTrainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        if isinstance(inputs, dict) and "sample_idx" in inputs:
            inputs = dict(inputs)
            inputs.pop("sample_idx")
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

class TrainingLossPlateauCallback(TrainerCallback):
    def __init__(self, optimizer, patience=PLATEAU_PATIENCE, factor=PLATEAU_FACTOR, min_lr=PLATEAU_MIN_LR):
        self.optimizer = optimizer
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.best = float("inf")
        self.window = deque(maxlen=patience)
        self.count = 0

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs is None: return
        loss = logs.get("loss")
        if loss is None: return
        self.window.append(loss)
        avg = float(np.mean(self.window))
        if avg + 1e-6 < self.best:
            self.best = avg
            self.count = 0
        else:
            self.count += 1
        if self.count >= self.patience:
            for g in self.optimizer.param_groups:
                old = g["lr"]
                new = max(self.min_lr, old * self.factor)
                if new < old:
                    g["lr"] = new
            print(f"📉 Plateau detected: reduced LR. New LR example: {self.optimizer.param_groups[0]['lr']}")
            self.count = 0
            self.window.clear()


# ----------------------------
# Load processor & model
# ----------------------------
def load_processor_and_model_with_fallback(local_cache_path, hf_model_name="facebook/seamless-m4t-v2-large", hf_token=None, device="cuda"):
    os.makedirs(local_cache_path, exist_ok=True)
    processor = None
    model = None

    try:
        print(f"📥 Trying local cache: {local_cache_path}")
        processor = SeamlessM4TProcessor.from_pretrained(local_cache_path)
        model = SeamlessM4TForSpeechToText.from_pretrained(local_cache_path)
        model.to(device)
        print("✅ Loaded from local cache.")
        return processor, model
    except Exception as e:
        print(f"⚠️ Local cache load failed: {e}")

    if hf_token is None:
        raise RuntimeError(f"❌ Local cache not found at {local_cache_path}. Cannot download from HF without a token.")

    print(f"🌐 Downloading from HF: {hf_model_name}")
    processor = SeamlessM4TProcessor.from_pretrained(hf_model_name, use_auth_token=hf_token)
    model = SeamlessM4TForSpeechToText.from_pretrained(hf_model_name, use_auth_token=hf_token)
    model.to(device)
    print("✅ Downloaded from HF.")
    processor.save_pretrained(local_cache_path)
    model.save_pretrained(local_cache_path)
    print(f"💾 Saved to local cache: {local_cache_path}")
    return processor, model


# ----------------------------
# Final evaluation
# ----------------------------
def run_full_audio_eval(model, processor, val_ds, device, dataset_name="val"):
    if len(val_ds) == 0: return {"bleu": 0.0, "chrf++": 0.0}
    model.eval(); torch.cuda.empty_cache()
    val_loader = DataLoader(val_ds, batch_size=1, collate_fn=collate_fn, shuffle=False, num_workers=2)
    all_segment_preds, all_segment_refs, all_segment_sample_idx = [], [], []
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    gen_kwargs = {"max_new_tokens": MAX_TARGET_LEN, "num_beams": 1, "no_repeat_ngram_size":3, "repetition_penalty":1.2, "tgt_lang":TARGET_LANG}
    with torch.inference_mode(), tqdm(total=len(val_loader), desc=f"Generating on {dataset_name}", unit="batch") as pbar:
        for batch in val_loader:
            sample_idx_tensor = batch.pop("sample_idx")
            batch = {k: v.to(device) for k,v in batch.items() if isinstance(v, torch.Tensor)}
            try:
                generated_ids = model.generate(**batch, **gen_kwargs)
            except TypeError:
                generated_ids = model.generate(**batch)
            decoded_preds = processor.batch_decode(generated_ids if isinstance(generated_ids, torch.Tensor) else generated_ids[0], skip_special_tokens=True)
            labels = batch.get("labels")
            if labels is not None:
                labels_np = labels.cpu().numpy()
                labels_clean = np.where(labels_np==-100, pad_id, labels_np)
                decoded_labels = processor.batch_decode(labels_clean, skip_special_tokens=True)
            else:
                decoded_labels = [""]*len(decoded_preds)
            all_segment_preds.extend(decoded_preds)
            all_segment_refs.extend(decoded_labels)
            all_segment_sample_idx.extend(sample_idx_tensor.cpu().numpy().tolist())
            del generated_ids
            for v in batch.values(): del v
            torch.cuda.empty_cache(); gc.collect()
            pbar.update(1)
    full_preds, full_refs = merge_segment_preds_to_full(all_segment_preds, all_segment_refs, all_segment_sample_idx)
    metrics = compute_metrics(full_preds, full_refs)
    eval_dir = os.path.join(OUTPUT_DIR, f"evaluation_{dataset_name}")
    os.makedirs(eval_dir, exist_ok=True)
    with open(os.path.join(eval_dir,f"predictions_{dataset_name}.txt"),"w",encoding="utf-8") as f: f.write("\n".join(full_preds))
    with open(os.path.join(eval_dir,f"references_{dataset_name}.txt"),"w",encoding="utf-8") as f: f.write("\n".join(full_refs))
    with open(os.path.join(eval_dir,f"metrics_{dataset_name}.json"),"w",encoding="utf-8") as f: json.dump(metrics,f,ensure_ascii=False,indent=2)
    print(f"✅ {dataset_name} metrics:", metrics)
    return metrics


# ----------------------------
# MAIN
# ----------------------------
def main(token=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    HF_TOKEN = os.getenv("HF_TOKEN") or token
    if HF_TOKEN is None or HF_TOKEN.strip() == "":
        raise RuntimeError("❌ HF_TOKEN environment variable not set! Please export your Hugging Face token.")

    processor, model = load_processor_and_model_with_fallback(
        local_cache_path=f"{OUTPUT_DIR}/cache_seamless",
        hf_model_name="facebook/seamless-m4t-v2-large",
        hf_token=HF_TOKEN,
        device=device
    )

    # ---------------- LoRA
    lora_config = LoraConfig(
        task_type="SEQ_2_SEQ_LM",
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # ---------------- Load datasets (all of them)
    TRAIN_FILES = [OLD_TRAIN]
    if os.path.exists(NEW_TRAIN_DIR):
        TRAIN_FILES += sorted(glob(os.path.join(NEW_TRAIN_DIR, "train_chunk*.pt")))
    print("📦 Train files (will be indexed):", TRAIN_FILES)

    VAL_FILES = [OLD_VAL, PART2_VAL]
    print("📦 Val files (will be indexed):", VAL_FILES)

    # Create datasets
    train_dataset = LazyChunkedPTDataset(TRAIN_FILES, processor)
    val_dataset = LazyChunkedPTDataset(VAL_FILES, processor)

    # ---------------- Training args
    training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BS,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_ratio=WARMUP_RATIO,
    num_train_epochs=EPOCHS,
    fp16=USE_FP16,
    logging_steps=LOGGING_STEPS,
    save_strategy="steps",
    save_steps=2000,
    save_total_limit=SAVE_TOTAL_LIMIT,
    report_to=REPORT_TO,
    eval_strategy="no",               # still no eval
    load_best_model_at_end=False,     # cannot load best model since no val metric
    metric_for_best_model=None,       # no metric needed
    )


    trainer = SafeSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        tokenizer=processor,
        data_collator=collate_fn,
        callbacks=[TrainingLossPlateauCallback(optimizer)],
    )

    # ---------------- Train
    trainer.train()

    # ---------------- Final Evaluation
    print("🏁 Training finished. Running final evaluation on validation datasets...")
    run_full_audio_eval(model, processor, val_dataset, device, dataset_name="final_val")

if __name__ == "__main__":
    main()
