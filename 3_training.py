# finetune_seamless_m4t_lora_full_audio_metrics_fixed_v5.py
import os
import json
import random
from glob import glob
import warnings
import sacrebleu
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm

from transformers import (
    SeamlessM4TProcessor,
    SeamlessM4TForSpeechToText,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
    GenerationConfig,
)
from peft import LoraConfig, get_peft_model

# ----------------------------
# SETTINGS
# ----------------------------
OLD_TRAIN = "./large_quality_data/train_dataset.pt"
OLD_VAL = "./large_quality_data/val_dataset.pt"
NEW_TRAIN_DIR = "./part-2_dataset"
OUTPUT_DIR = "./3rd_finetune"

SEED = 42
SAMPLE_RATE = 16000
CHUNK_SECS = 15
OVERLAP_SECS = 1
CHUNK_LEN = CHUNK_SECS * SAMPLE_RATE
OVERLAP = OVERLAP_SECS * SAMPLE_RATE

BATCH_SIZE = 4
GRAD_ACCUM = 4
LEARNING_RATE = 1e-5
EPOCHS = 30
WEIGHT_DECAY = 0.01

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2"]

MAX_TARGET_LEN = 128
FP16 = True
LOGGING_STEPS = 50
SAVE_STEPS = 1000
EVAL_STEPS = 500
REPORT_TO = None
TARGET_LANG = "eng"  # target language for generation

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# Reproducibility
# ----------------------------
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ----------------------------
# Suppress warnings
# ----------------------------
warnings.filterwarnings("ignore", message=".*Token indices sequence length is longer.*")
warnings.filterwarnings("ignore", message=".*was truncated to a.*")

# ----------------------------
# Dataset
# ----------------------------
class ChunkedPTDataset(Dataset):
    def __init__(self, pt_files, processor, chunk_len=CHUNK_LEN, overlap=OVERLAP, sr=SAMPLE_RATE, max_target_len=MAX_TARGET_LEN):
        self.processor = processor
        self.chunk_len = int(chunk_len)
        self.overlap = int(overlap)
        self.sr = sr
        self.max_target_len = max_target_len

        self.samples = []
        for f in pt_files:
            if os.path.exists(f):
                data = torch.load(f)
                if isinstance(data, (list, tuple)):
                    self.samples.extend(data)
                elif isinstance(data, dict):
                    self.samples.append(data)

        self.segments = []
        for s_idx, s in enumerate(tqdm(self.samples, desc="Indexing samples", unit="sample")):
            audio = s.get("audio_array") if s.get("audio_array") is not None else s.get("audio")
            if audio is None:
                continue
            if isinstance(audio, torch.Tensor):
                audio = audio.cpu().numpy()
            length = len(audio)
            if length <= 0:
                continue
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

        print(f"📦 Loaded {len(self.samples)} raw samples -> {len(self.segments)} segments total")

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        s_idx, start, end = self.segments[idx]
        s = self.samples[s_idx]

        audio = s.get("audio_array") if s.get("audio_array") is not None else s.get("audio")
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

        label_enc = self.processor(
            text=text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_target_len,
        ).input_ids

        input_features = inputs.input_features.squeeze(0)
        attention_mask = inputs.attention_mask.squeeze(0) if hasattr(inputs, "attention_mask") else None
        labels = label_enc.squeeze(0)

        return {"input_features": input_features, "labels": labels, "sample_idx": s_idx, "attention_mask": attention_mask}

# ----------------------------
# Collate
# ----------------------------
def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return {"input_features": torch.zeros(1, 100, 80),
                "labels": torch.zeros(1, dtype=torch.long),
                "sample_idx": torch.zeros(1, dtype=torch.long)}

    input_feats = nn.utils.rnn.pad_sequence([b["input_features"] for b in batch], batch_first=True, padding_value=0.0)
    labels = nn.utils.rnn.pad_sequence([b["labels"] for b in batch], batch_first=True, padding_value=-100)
    sample_idx = torch.tensor([b["sample_idx"] for b in batch], dtype=torch.long)

    out = {"input_features": input_feats, "labels": labels, "sample_idx": sample_idx}
    if "attention_mask" in batch[0] and batch[0]["attention_mask"] is not None:
        masks = nn.utils.rnn.pad_sequence([b["attention_mask"] for b in batch], batch_first=True, padding_value=0)
        out["attention_mask"] = masks
    return out

# ----------------------------
# Metrics
# ----------------------------
def compute_metrics(preds, refs):
    bleu = sacrebleu.corpus_bleu(preds, [refs])
    chrf = sacrebleu.corpus_chrf(preds, [refs])
    return {"bleu": bleu.score, "chrf++": chrf.score}

def compute_full_audio_metrics(eval_preds, processor):
    pred_ids, label_ids, sample_idx = eval_preds

    if isinstance(pred_ids, tuple):
        pred_arr = pred_ids[0]
    else:
        pred_arr = pred_ids

    if isinstance(pred_arr, torch.Tensor):
        pred_arr = pred_arr.detach().cpu().numpy()
    if isinstance(label_ids, torch.Tensor):
        label_ids = label_ids.detach().cpu().numpy()
    sample_idx = np.array(sample_idx)

    pad_id = processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else processor.tokenizer.eos_token_id
    label_ids_clean = np.where(label_ids == -100, pad_id, label_ids)

    decoded_preds = processor.batch_decode(pred_arr, skip_special_tokens=True)
    decoded_labels = processor.batch_decode(label_ids_clean, skip_special_tokens=True)

    merged_preds = {}
    merged_refs = {}
    for idx, s_idx in enumerate(sample_idx):
        merged_preds.setdefault(int(s_idx), []).append(decoded_preds[idx])
        merged_refs.setdefault(int(s_idx), []).append(decoded_labels[idx])

    full_preds = [" ".join(merged_preds[k]) for k in sorted(merged_preds.keys())]
    full_refs = [" ".join(merged_refs[k]) for k in sorted(merged_refs.keys())]

    return compute_metrics(full_preds, full_refs)

# ----------------------------
# Custom Trainer
# ----------------------------
class SafeSeq2SeqTrainer(Seq2SeqTrainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        if "sample_idx" in inputs:
            inputs.pop("sample_idx")
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

# ----------------------------
# MAIN
# ----------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Device: {device}")

    processor = SeamlessM4TProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
    model = SeamlessM4TForSpeechToText.from_pretrained("facebook/seamless-m4t-v2-large")

    if not hasattr(model, "generation_config") or model.generation_config is None:
        model.generation_config = GenerationConfig()
    setattr(model.generation_config, "tgt_lang", TARGET_LANG)
    print(f"🌐 Target generation language set to: {TARGET_LANG}")

    # LoRA setup
    lora_config = LoraConfig(
        task_type="SEQ_2_SEQ_LM",
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none"
    )
    model = get_peft_model(model, lora_config)
    print("✅ LoRA adapters attached")

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.to(device)

    # Datasets
    train_files = [OLD_TRAIN] + sorted(glob(os.path.join(NEW_TRAIN_DIR, "train_chunk*.pt")))
    val_files = [OLD_VAL] + sorted(glob(os.path.join(NEW_TRAIN_DIR, "val_chunk*.pt")))

    print(f"📦 Train files: {train_files}")
    print(f"📦 Val files: {val_files}")

    train_ds = ChunkedPTDataset(train_files, processor)
    val_ds = ChunkedPTDataset(val_files, processor)

    # Mixed precision flags
    use_bf16 = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
    use_fp16 = torch.cuda.is_available() and not use_bf16 and FP16

    # ----------------------------
    # Training Arguments
    # ----------------------------
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        num_train_epochs=EPOCHS,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        eval_steps=EVAL_STEPS,
        eval_strategy="steps",
        predict_with_generate=False,  # critical: speed up training
        fp16=use_fp16,
        bf16=use_bf16,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        report_to=REPORT_TO,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    # ----------------------------
    # Metrics wrapper
    # ----------------------------
    def trainer_compute_metrics(eval_preds):
        preds = getattr(eval_preds, "predictions", None)
        labels = getattr(eval_preds, "label_ids", None)
        sample_idx = list(range(labels.shape[0])) if labels is not None else []
        return compute_full_audio_metrics((preds, labels, sample_idx), processor)

    # ----------------------------
    # Trainer
    # ----------------------------
    trainer = SafeSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        compute_metrics=trainer_compute_metrics,
    )

    print("🚀 Starting training...")
    trainer.train()

    # Save final model
    final_dir = os.path.join(OUTPUT_DIR, "final_model_3_")
    os.makedirs(final_dir, exist_ok=True)
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)

    # Final evaluation
    print("🔎 Running final evaluation...")
    eval_res = trainer.evaluate()
    with open(os.path.join(final_dir, "final_eval_results.json"), "w") as f:
        json.dump(eval_res, f, indent=2)

    print("✅ Training and evaluation complete.")

if __name__ == "__main__":
    main()
