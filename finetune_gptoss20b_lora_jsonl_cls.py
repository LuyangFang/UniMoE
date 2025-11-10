#!/usr/bin/env python
# Fast multi-label LoRA finetune for openai/gpt-oss-20b on TABULAR CSV/TSV.
# Improvements:
# - Mean pooling + LayerNorm + 2-layer MLP classifier
# - Class imbalance via pos_weight
# - Train/Val split, EarlyStopping, restore best epoch (via adapters+head file on disk)
# - Per-label threshold tuning on Val; apply to Test
# - Optimizer param groups (head 5x LR)
# - Raw text tokenization (no chat template)
# - Collator pad_to_multiple_of=8
# - Writes:
#   * <outdir>/train_probs.csv (preds/probs per label on TRAIN)
#   * <outdir>/train_emb_lowdim.csv (low-dim pooled embeddings on TRAIN)
#   * <outdir>/best_adapters_head.pt (only LoRA adapters + classifier head)

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Helps avoid fragmentation-related OOM at large scales
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import random
import csv
import copy
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModel,                 # faster backbone-only forward
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    TrainerCallback,
)
from transformers import Mxfp4Config
from peft import LoraConfig, get_peft_model, TaskType
from peft.utils import get_peft_model_state_dict

# new: metrics for kappa/F1@0.5
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score


# ---------------------------
# Utilities
# ---------------------------

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lora_target_substrings() -> List[str]:
    # Focus LoRA on core projection modules (avoid overly broad matches)
    return [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "up_proj", "down_proj",
        # leave out very broad 'dense'/'proj' catch-alls
    ]


# ---------------------------
# Data loading (tabular multi-label)
# ---------------------------

def _normalize_colname(name: str) -> str:
    return str(name).strip()

def _is_group_column(name: str) -> bool:
    return _normalize_colname(name).lower() == "group"

def read_tabular_file(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")  # autodetects CSV/TSV
    df = df.dropna(axis=1, how="all")
    df.columns = [_normalize_colname(c) for c in df.columns]
    return df

def unify_label_columns(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], str]:
    if train_df.shape[1] < 2 or test_df.shape[1] < 2:
        raise ValueError("Expected at least 2 columns: [text, >=1 label].")

    text_col = train_df.columns[0]  # first column is text

    def label_candidates(df: pd.DataFrame) -> List[str]:
        return [c for c in df.columns[1:] if not _is_group_column(c)]

    train_labels = label_candidates(train_df)
    test_labels = label_candidates(test_df)

    # union preserving train order then test extras
    label_cols, seen = [], set()
    for c in train_labels + [c for c in test_labels if c not in train_labels]:
        if c not in seen:
            label_cols.append(c)
            seen.add(c)

    for df in (train_df, test_df):
        for c in label_cols:
            if c not in df.columns:
                df[c] = 0
        # drop Group if present
        drop_cols = [c for c in df.columns if _is_group_column(c)]
        if drop_cols:
            df.drop(columns=drop_cols, inplace=True, errors="ignore")
        # force 0/1 ints for labels
        for c in label_cols:
            s = df[c]
            if s.dtype == object:
                m = s.astype(str).str.strip().str.lower()
                df[c] = np.where(m.isin(["1", "true", "yes", "y", "t"]), 1,
                                 np.where(m.isin(["0", "false", "no", "n", "f", "", "nan", "none"]), 0, 0))
            else:
                df[c] = pd.to_numeric(s, errors="coerce").fillna(0).astype(int)
            df[c] = df[c].clip(0, 1)

    return train_df, test_df, label_cols, text_col

def build_hf_datasets_from_frames(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_cols: List[str],
    text_col: str
) -> Tuple[DatasetDict, List[str], List[str]]:
    def to_dataset(df: pd.DataFrame):
        texts = df[text_col].astype(str).fillna("").tolist()
        labels = df[label_cols].astype(int).values.tolist()
        ds = Dataset.from_dict({"text": texts, "labels": labels})
        return ds, texts
    train_ds, train_texts = to_dataset(train_df)
    test_ds, test_texts = to_dataset(test_df)
    return DatasetDict({"train": train_ds, "test": test_ds}), train_texts, test_texts


# ---------------------------
# Model wrapper for classification (fast path)
# ---------------------------

class CausalLMWithClassifier(torch.nn.Module):
    """
    Uses AutoModel (transformer backbone only) -> masked mean pooling -> LN -> MLP -> logits
    Multi-label: BCEWithLogits with pos_weight (for imbalance).
    Exposes optional low-dimensional embedding head for export.
    """
    def __init__(self, base_model: AutoModel, hidden_size: int, num_labels: int,
                 pos_weight: Optional[torch.Tensor] = None, embed_dim: int = 128):
        super().__init__()
        self.base = base_model
        self.num_labels = num_labels
        self.norm = torch.nn.LayerNorm(hidden_size)
        # low-dimensional embedding head (after pooling+norm)
        self.embed_low = torch.nn.Linear(hidden_size, embed_dim, bias=False)
        self.mlp = torch.nn.Sequential(
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.GELU(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(hidden_size // 2, num_labels),
        )
        # store pos_weight as buffer to follow device
        if pos_weight is None:
            pos_weight = torch.ones(num_labels, dtype=torch.float32)
        self.register_buffer("pos_weight", pos_weight, persistent=False)

    # Pass-through gradient checkpointing
    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.base, "gradient_checkpointing_enable"):
            return self.base.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        if hasattr(self.base, "gradient_checkpointing_disable"):
            return self.base.gradient_checkpointing_disable()

    def _masked_mean_pool(self, last_hidden, attention_mask):
        # last_hidden: (B, T, H), attention_mask: (B, T)
        if attention_mask is None:
            return last_hidden.mean(dim=1)
        mask = attention_mask.unsqueeze(-1).float()  # (B, T, 1)
        summed = (last_hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return summed / denom

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                return_hidden: bool=False, **kwargs):
        out = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
            output_hidden_states=return_hidden or getattr(self.base.config, "output_hidden_states", False),
        )
        last = out.last_hidden_state  # (B, T, H)
        pooled = self._masked_mean_pool(last, attention_mask)
        pooled = self.norm(pooled)
        logits = self.mlp(pooled)
        lowdim = self.embed_low(pooled)  # (B, embed_dim)

        loss = None
        if labels is not None:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits.float(), labels.float(), pos_weight=self.pos_weight.to(logits.device)
            )

        ret = {"loss": loss, "logits": logits, "lowdim": lowdim}
        if return_hidden:
            ret["hidden_states"] = out.hidden_states
        return ret


# ---------------------------
# Collator with pad_to_multiple_of=8 (Tensor Cores)
# ---------------------------

@dataclass
class Collator(DataCollatorWithPadding):
    def __call__(self, features):
        labels = [f.pop("labels") for f in features]
        batch = super().__call__(features)  # pad_to_multiple_of handled by base
        batch["labels"] = torch.tensor(labels, dtype=torch.float32)
        return batch


# ---------------------------
# Metrics & threshold tuning
# ---------------------------

def multilabel_metrics_from_logits(labels: torch.Tensor, logits: torch.Tensor, thresholds) -> Dict[str, float]:
    with torch.no_grad():
        probs = torch.sigmoid(logits.float())
        if isinstance(thresholds, float):
            preds = (probs >= thresholds).to(torch.int64)
        else:
            thr = torch.tensor(thresholds, device=probs.device, dtype=probs.dtype).view(1, -1)
            preds = (probs >= thr).to(torch.int64)
        y_true = labels.to(torch.int64)

    y_true_np = y_true.cpu().numpy()
    y_pred_np = preds.cpu().numpy()

    exact_match = float((y_pred_np == y_true_np).all(axis=1).mean()) if y_true_np.size > 0 else 0.0

    TP = int(np.logical_and(y_pred_np == 1, y_true_np == 1).sum())
    FP = int(np.logical_and(y_pred_np == 1, y_true_np == 0).sum())
    FN = int(np.logical_and(y_pred_np == 0, y_true_np == 1).sum())

    micro_prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    micro_rec  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    micro_f1   = (2 * micro_prec * micro_rec / (micro_prec + micro_rec)) if (micro_prec + micro_rec) > 0 else 0.0

    C = y_true_np.shape[1] if y_true_np.ndim == 2 else 0
    f1_per_label = []
    if C > 0:
        for c in range(C):
            yt = y_true_np[:, c]
            yp = y_pred_np[:, c]
            TPc = int(np.logical_and(yp == 1, yt == 1).sum())
            FPc = int(np.logical_and(yp == 1, yt == 0).sum())
            FNc = int(np.logical_and(yp == 0, yt == 1).sum())
            prec_c = TPc / (TPc + FPc) if (TPc + FPc) > 0 else 0.0
            rec_c  = TPc / (TPc + FNc) if (TPc + FNc) > 0 else 0.0
            f1_c   = (2 * prec_c * rec_c / (prec_c + rec_c)) if (prec_c + rec_c) > 0 else 0.0
            f1_per_label.append(f1_c)
    macro_f1 = float(np.mean(f1_per_label)) if f1_per_label else 0.0

    return {
        "exact_match": exact_match,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
    }

def find_per_label_thresholds(model, dataloader, device, n_labels):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            labels = batch.pop("labels").to(device)
            feed = {k: v.to(device) for k, v in batch.items() if k in ("input_ids","attention_mask")}
            logits = model(**feed)["logits"]
            all_logits.append(logits)
            all_labels.append(labels)
    L = torch.cat(all_labels, 0).float()
    Z = torch.cat(all_logits, 0).float()
    P = torch.sigmoid(Z).cpu().numpy()
    Y = L.cpu().numpy().astype(int)

    thresholds = np.full(n_labels, 0.5, dtype=np.float32)
    for j in range(n_labels):
        best_f1, best_t = 0.0, 0.5
        for t in np.linspace(0.05, 0.95, 19):
            yp = (P[:, j] >= t).astype(int)
            tp = ((yp == 1) & (Y[:, j] == 1)).sum()
            fp = ((yp == 1) & (Y[:, j] == 0)).sum()
            fn = ((yp == 0) & (Y[:, j] == 1)).sum()
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1   = (2*prec*rec/(prec+rec)) if (prec+rec) > 0 else 0.0
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[j] = best_t
    return thresholds


# ---------------------------
# Validation callback (best epoch via on-disk adapters+head + early stopping)
# ---------------------------

class ValEvalCallback(TrainerCallback):
    def __init__(self, val_loader, device, label_names: List[str], patience: int = 3, outdir: str = "."):
        super().__init__()
        self.val_loader = val_loader
        self.device = device
        self.label_names = label_names
        self.best_exact = 0.0
        self.best_epoch = -1
        self.per_epoch = []
        self.patience = patience
        self.bad_epochs = 0
        self.best_path = os.path.join(outdir, "best_adapters_head.pt")  # only adapters + head

    @torch.no_grad()
    def _snapshot_trainable_to_disk(self, model):
        # LoRA adapters only
        try:
            lora_sd = get_peft_model_state_dict(model.base)
        except Exception:
            lora_sd = {k: v for k, v in model.state_dict().items() if "lora_" in k}

        # Classifier head + LayerNorm + embed_low
        head_sd = {
            k: v for k, v in model.state_dict().items()
            if k.startswith("mlp.") or k.startswith("norm.") or k.startswith("embed_low.")
        }

        # Move to CPU just for serialization (no persistent RAM copy kept)
        snap = {k: v.detach().cpu() for k, v in {**lora_sd, **head_sd}.items()}
        tmp_path = self.best_path + ".tmp"
        torch.save(snap, tmp_path)
        os.replace(tmp_path, self.best_path)  # atomic rename
        del snap
        torch.cuda.empty_cache()

    @torch.no_grad()
    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs["model"]
        model.eval()

        all_labels, all_logits = [], []
        for batch in self.val_loader:
            labels = batch.pop("labels").to(self.device)
            feed = {k: v.to(self.device) for k, v in batch.items() if k in ("input_ids", "attention_mask")}
            logits = model(**feed)["logits"]

            # keep tensors on CPU to reduce VRAM
            all_labels.append(labels.cpu())
            all_logits.append(logits.cpu())

        labels_all = torch.cat(all_labels, dim=0)
        logits_all = torch.cat(all_logits, dim=0)

        # use scalar 0.5 for model selection; thresholds tuned later
        metrics = multilabel_metrics_from_logits(labels_all, logits_all, thresholds=0.5)
        exact = metrics["exact_match"]
        micro = metrics["micro_f1"]
        self.per_epoch.append((float(state.epoch), exact, micro))

        improved = exact > self.best_exact
        if improved:
            self.best_exact = exact
            self.best_epoch = int(round(state.epoch))
            self.bad_epochs = 0
            # save only adapters + head to disk
            self._snapshot_trainable_to_disk(model)
        else:
            self.bad_epochs += 1

        print(
            f"\n[VAL] Epoch {state.epoch:.2f} — N: {labels_all.size(0)} | "
            f"ExactMatch: {exact:.4f} | Micro-F1: {micro:.4f} | "
            f"BestExact: {self.best_exact:.4f} (epoch {self.best_epoch})",
            flush=True
        )

        if self.bad_epochs >= self.patience:
            print(f"[EARLY-STOP] Patience reached ({self.patience}) — will stop after this epoch.", flush=True)
            control.should_training_stop = True

        torch.cuda.empty_cache()


# ---------------------------
# Training
# ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True, help="Path to train CSV/TSV (first col=text; others=labels)")
    parser.add_argument("--test_file", type=str, required=True, help="Path to test CSV/TSV (first col=text; others=labels)")
    parser.add_argument("--model_name", type=str, default="openai/gpt-oss-20b")
    parser.add_argument("--output_dir", type=str, default="./no_save")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accum_steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--log_every_steps", type=int, default=500)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=0.3)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.95)
    parser.add_argument("--threshold", type=float, default=0.5, help="fallback/global threshold if per-label not used")
    parser.add_argument("--probs_out", type=str, default=None)
    parser.add_argument("--include_text_in_csv", action="store_true")
    parser.add_argument("--early_stop_patience", type=int, default=3)
    parser.add_argument("--embed_dim", type=int, default=128, help="Low-dimensional embedding size to export for TRAIN")
    args = parser.parse_args()

    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ----- Load data -----
    train_df = read_tabular_file(args.train_file)
    test_df  = read_tabular_file(args.test_file)
    train_df, test_df, label_cols, text_col = unify_label_columns(train_df, test_df)
    num_labels = len(label_cols)
    if num_labels < 1:
        raise ValueError("No label columns found after first column.")

    # Build HF datasets
    ds_raw, train_texts, test_texts = build_hf_datasets_from_frames(train_df, test_df, label_cols, text_col)

    # Split train into train/val for thresholding + early stopping
    raw_train = ds_raw["train"]
    val_ratio = 0.1
    split = raw_train.train_test_split(test_size=val_ratio, seed=args.seed, shuffle=True)
    ds_raw = DatasetDict({"train": split["train"], "val": split["test"], "test": ds_raw["test"]})

    # ----- Tokenizer -----
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def preprocess(ex):
        enc = tok(
            ex["text"],
            max_length=args.max_length,
            truncation=True,
            padding=False
        )
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": ex["labels"]}

    ds_proc = DatasetDict({
        "train": ds_raw["train"].map(preprocess, remove_columns=ds_raw["train"].column_names, desc="Tokenizing train"),
        "val":   ds_raw["val"].map(preprocess,   remove_columns=ds_raw["val"].column_names,   desc="Tokenizing val"),
        "test":  ds_raw["test"].map(preprocess,  remove_columns=ds_raw["test"].column_names,  desc="Tokenizing test"),
    })

    # ----- Compute pos_weight from TRAIN only -----
    train_arr = np.array(ds_raw["train"]["labels"], dtype=np.int64)
    pos = train_arr.sum(axis=0).astype(np.float32)
    neg = (train_arr.shape[0] - pos).astype(np.float32)
    pos_weight_vec = torch.tensor(np.where(pos > 0, neg / pos, 1.0), dtype=torch.float32)
    print("[INFO] pos_weight per label:", dict(zip(label_cols, pos_weight_vec.tolist())))

    # ----- Config & attention impl -----
    torch_dtype = torch.bfloat16 if (args.bf16 or torch.cuda.is_bf16_supported()) else torch.float16
    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    if hasattr(config, "use_cache"):
        config.use_cache = False
    # Do NOT force output_hidden_states globally; we’ll request them only when needed.
    if hasattr(config, "attn_implementation"):
        for impl in ("flash_attention_2", "sdpa", "eager"):
            try:
                config.attn_implementation = impl
                break
            except Exception:
                pass

    # ----- Base model (encoder-only forward) -----
    base = AutoModel.from_pretrained(
        args.model_name,
        config=config,
        trust_remote_code=True,
        device_map="cuda",
        torch_dtype=torch_dtype,
        quantization_config=Mxfp4Config(dequantize=True),
    )

    # Gradient checkpointing on backbone
    if args.gradient_checkpointing and hasattr(base, "gradient_checkpointing_enable"):
        base.gradient_checkpointing_enable()
        if hasattr(base.config, "use_cache"):
            base.config.use_cache = False

    # ----- LoRA -----
    target_modules = lora_target_substrings()
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,  # valid for decoder stacks when using AutoModel
        target_modules=target_modules,
    )
    base = get_peft_model(base, lora_cfg)
    base.print_trainable_parameters()

    # ----- Classifier + low-dim embedding head -----
    hidden_size = getattr(base.config, "hidden_size", None) or getattr(base.config, "n_embd", None)
    if hidden_size is None:
        hidden_size = base.get_input_embeddings().embedding_dim
    model = CausalLMWithClassifier(
        base,
        hidden_size=hidden_size,
        num_labels=num_labels,
        pos_weight=pos_weight_vec,
        embed_dim=args.embed_dim,
    )

    # Keep head in fp32 (classifier, norm, low-dim head)
    for p in list(model.mlp.parameters()) + list(model.norm.parameters()) + list(model.embed_low.parameters()):
        p.requires_grad = True
        p.data = p.data.to(torch.float32)

    # ----- Collator -----
    collator = Collator(tokenizer=tok, padding=True, pad_to_multiple_of=8)

    # ----- Precision flags -----
    do_bf16 = args.bf16 or torch.cuda.is_bf16_supported()
    do_fp16 = (not do_bf16) and (args.fp16 or torch.cuda.is_available())

    # ----- Trainer with tidy prints + custom optimizer groups -----
    class ClassificationTrainer(Trainer):
        def save_model(self, *a, **k): return
        def _save_checkpoint(self, *a, **k): return
        _mem_reset_done = False

        def _gpu_index(self):
            try:
                dev = getattr(self.args, "device", None)
                if isinstance(dev, torch.device) and dev.index is not None:
                    return dev.index
            except Exception:
                pass
            try:
                return torch.cuda.current_device()
            except Exception:
                return 0

        def _gpu_mem_line(self):
            if not torch.cuda.is_available():
                return "mem n/a"
            idx = self._gpu_index()
            try:
                torch.cuda.synchronize(idx)
            except Exception:
                pass
            alloc = torch.cuda.memory_allocated(idx)
            reserved = torch.cuda.memory_reserved(idx)
            peak = torch.cuda.max_memory_allocated(idx)
            total = torch.cuda.get_device_properties(idx).total_memory
            to_gb = lambda b: b / (1024**3)
            return f"mem {to_gb(alloc):.1f}G alloc / {to_gb(reserved):.1f}G res / {to_gb(total):.1f}G tot (peak {to_gb(peak):.1f}G)"

        def _format_train_line(self, step, loss_val, exact_val, epoch, lr, mem):
            return (f"[TRAIN] epoch {epoch:>4.2f} | step {step:>6d} | "
                    f"loss {loss_val:>7.4f} | exact {exact_val:>5.2f} | lr {lr:.2e} | {mem}")

        def create_optimizer(self):
            if self.optimizer is None:
                head_params = list(model.mlp.parameters()) + list(model.norm.parameters())
                lora_params = [p for n,p in model.named_parameters() if p.requires_grad and 'lora_' in n]
                base_others = [p for n,p in model.named_parameters()
                               if (p.requires_grad and 'lora_' not in n and 'mlp' not in n and 'norm' not in n)]
                grouped = [
                    {"params": head_params, "lr": self.args.learning_rate * 5.0, "weight_decay": self.args.weight_decay},
                    {"params": lora_params, "lr": self.args.learning_rate, "weight_decay": self.args.weight_decay},
                    {"params": base_others, "lr": self.args.learning_rate, "weight_decay": self.args.weight_decay},
                ]
                self.optimizer = torch.optim.AdamW(
                    grouped,
                    betas=(self.args.adam_beta1, self.args.adam_beta2),
                    lr=self.args.learning_rate,
                    weight_decay=self.args.weight_decay
                )
            return self.optimizer

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.get("labels")
            feed = {k: v for k, v in inputs.items() if k in ("input_ids", "attention_mask")}
            outputs = model(**feed, labels=labels)
            loss = outputs["loss"]

            step = int(self.state.global_step) if self.state.global_step is not None else 0
            gate = max(1, getattr(self.args, "_log_every_steps", 500))

            if torch.cuda.is_available() and not self._mem_reset_done and step == 0:
                try:
                    torch.cuda.reset_peak_memory_stats(self._gpu_index())
                    self._mem_reset_done = True
                except Exception:
                    pass

            lr = float("nan")
            try:
                if self.optimizer and len(self.optimizer.param_groups) > 0:
                    lr = float(self.optimizer.param_groups[0].get("lr", float("nan")))
            except Exception:
                pass

            if step == 0 or (step % gate == 0):
                try:
                    with torch.no_grad():
                        logits = outputs["logits"]
                        probs = torch.sigmoid(logits.float())
                        preds = (probs >= 0.5).to(torch.int64)  # scalar 0.5 during training logs
                        exact = (preds == labels.to(torch.int64)).all(dim=1).float().mean().item()
                    print(self._format_train_line(step, float(loss.detach().item()), exact,
                                                  float(self.state.epoch or 0.0), lr, self._gpu_mem_line()),
                          flush=True)
                except Exception:
                    pass

            return (loss, outputs) if return_outputs else loss

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size),
        gradient_accumulation_steps=args.accum_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        bf16=do_bf16,
        fp16=(not do_bf16) and do_fp16,
        dataloader_num_workers=4,
        report_to=[],
        max_grad_norm=args.max_grad_norm,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
    )
    setattr(training_args, "_log_every_steps", int(max(1, args.log_every_steps)))

    trainer = ClassificationTrainer(
        model=model,
        args=training_args,
        train_dataset=ds_proc["train"],
        eval_dataset=None,
        data_collator=collator,
        processing_class=tok,
    )

    # GPU summary
    if torch.cuda.is_available():
        try:
            idx = torch.cuda.current_device()
        except Exception:
            idx = 0
        props = torch.cuda.get_device_properties(idx)
        total_gb = props.total_memory / (1024**3)
        print(f"[GPU] Using device {idx}: {props.name} | {total_gb:.1f} GB total")

    # Val loader + callback (best epoch tracking + early stop)
    val_loader = trainer.get_eval_dataloader(ds_proc["val"])
    val_cb = ValEvalCallback(
        val_loader=val_loader,
        device=trainer.args.device,
        label_names=label_cols,
        patience=args.early_stop_patience,
        outdir=args.output_dir,
    )
    trainer.add_callback(val_cb)

    # Train
    print(f"[SETUP] Train size: {len(ds_proc['train'])} | Val size: {len(ds_proc['val'])} | "
          f"Test size: {len(ds_proc['test'])} | Labels: {num_labels} ({', '.join(map(str, label_cols))})")
    trainer.train()

    # Restore best epoch trainable weights (adapters + head) from disk
    best_path = os.path.join(args.output_dir, "best_adapters_head.pt")
    if os.path.isfile(best_path):
        ckpt = torch.load(best_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        print(f"[RESTORE] Loaded adapters+head from epoch {val_cb.best_epoch} "
              f"(val exact={val_cb.best_exact:.4f}); "
              f"missing={len(missing)}, unexpected={len(unexpected)}")
        del ckpt
    else:
        print("[RESTORE] No best_adapters_head.pt found; proceeding with last-epoch weights.")

    # Tune per-label thresholds on VAL using the best model
    per_label_thresholds = find_per_label_thresholds(model, val_loader, trainer.args.device, num_labels)
    thr_dict = dict(zip(label_cols, [float(f"{t:.3f}") for t in per_label_thresholds.tolist()]))
    print("[VAL] per-label thresholds:", thr_dict)

    # ----- Final Test: metrics (now also F1@0.5 and average kappa@0.5) -----
    test_loader = trainer.get_test_dataloader(ds_proc["test"])
    model.eval()
    all_labels, all_logits = [], []

    with torch.no_grad():
        for batch in test_loader:
            labels = batch.pop("labels").to(trainer.args.device)
            feed = {k: v.to(trainer.args.device) for k, v in batch.items() if k in ("input_ids","attention_mask")}
            out = model(**feed)  # logits + lowdim (we don't export test embeddings)
            logits = out["logits"]

            all_labels.append(labels.cpu())
            all_logits.append(logits.cpu())

    labels_all = torch.cat(all_labels, dim=0)
    logits_all = torch.cat(all_logits, dim=0)
    probs_all = torch.sigmoid(logits_all.float()).cpu().numpy()
    y_true_np = labels_all.cpu().numpy().astype(int)

    # exact/micro/macro using tuned per-label thresholds (kept as before)
    metrics_thresh = multilabel_metrics_from_logits(labels_all, logits_all, thresholds=per_label_thresholds)
    exact_match = metrics_thresh["exact_match"]

    # F1 and Kappa at hard 0.5 (as requested)
    preds_05 = (probs_all >= 0.5).astype(int)
    micro_f1_05 = float(f1_score(y_true_np, preds_05, average="micro", zero_division=0))
    macro_f1_05 = float(f1_score(y_true_np, preds_05, average="macro", zero_division=0))
    # average Cohen's kappa over labels
    kappas = []
    if y_true_np.size > 0:
        C = y_true_np.shape[1]
        for j in range(C):
            kappas.append(cohen_kappa_score(y_true_np[:, j], preds_05[:, j]))
    avg_kappa = float(np.mean(kappas)) if kappas else 0.0

    # Per-label accuracy at 0.5
    per_label_acc_vals = (preds_05 == y_true_np).mean(axis=0).tolist()
    per_label_accuracy = {label_cols[i]: float(per_label_acc_vals[i]) for i in range(len(label_cols))}

    print("\n[METRICS] Test set (best epoch)")
    print(f"  samples            : {y_true_np.shape[0]}")
    print(f"  exact_match (thr*) : {exact_match:.4f}   (*per-label tuned thresholds)")
    print(f"  micro_f1 @0.5      : {micro_f1_05:.4f}")
    print(f"  macro_f1 @0.5      : {macro_f1_05:.4f}")
    print(f"  avg_kappa @0.5     : {avg_kappa:.4f}")
    print("  per_label_accuracy @0.5 :")
    for lbl in label_cols:
        print(f"    {lbl:>15}: {per_label_accuracy[lbl]:.4f}")

    # ----- TRAIN EXPORTS: per-sample probabilities & low-dim embeddings -----
    train_export_loader = trainer.get_eval_dataloader(ds_proc["train"])  # eval-style loader (no shuffle)
    model.eval()
    tr_logits_list, tr_labels_list, tr_lowdim_list = [], [], []
    with torch.no_grad():
        for batch in train_export_loader:
            labels = batch.pop("labels").to(trainer.args.device)
            feed = {k: v.to(trainer.args.device) for k, v in batch.items() if k in ("input_ids","attention_mask")}
            out = model(**feed)  # includes logits + lowdim
            tr_logits_list.append(out["logits"].cpu())
            tr_lowdim_list.append(out["lowdim"].cpu())
            tr_labels_list.append(labels.cpu())

    tr_logits = torch.cat(tr_logits_list, dim=0)
    tr_probs = torch.sigmoid(tr_logits.float()).cpu().numpy()
    tr_labels = torch.cat(tr_labels_list, dim=0).cpu().numpy().astype(int)
    tr_preds05 = (tr_probs >= 0.5).astype(int)
    N_tr, C_tr = tr_labels.shape

    # TRAIN probabilities CSV (instead of TEST)
    probs_path = args.probs_out if args.probs_out else os.path.join(args.output_dir, "train_probs.csv")
    os.makedirs(os.path.dirname(probs_path), exist_ok=True)

    header = ["sample_index"]
    if args.include_text_in_csv:
        header.append("text")
    for lbl in label_cols:
        header.append(f"true_{lbl}")
    for lbl in label_cols:
        header.append(f"pred_{lbl}")
    for lbl in label_cols:
        header.append(f"prob_{lbl}")

    # Ensure we have train texts if requested
    train_texts_list = []
    if args.include_text_in_csv:
        try:
            train_texts_list = ds_raw["train"]["text"]
        except Exception:
            train_texts_list = []

    with open(probs_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(N_tr):
            row = [i]
            if args.include_text_in_csv:
                row.append(train_texts_list[i] if i < len(train_texts_list) else "")
            row.extend([int(tr_labels[i, j]) for j in range(C_tr)])
            row.extend([int(tr_preds05[i, j]) for j in range(C_tr)])
            row.extend([float(tr_probs[i, j]) for j in range(C_tr)])
            writer.writerow(row)

    # TRAIN low-dim embeddings CSV
    emb_lowdim = torch.cat(tr_lowdim_list, dim=0).numpy()   # (N_tr, D)
    emb_lowdim_path = os.path.join(args.output_dir, "train_emb_lowdim.csv")

    def write_emb_csv(path, arr):
        D = arr.shape[1]
        cols = ["sample_index"] + [f"e{j}" for j in range(D)]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for i in range(arr.shape[0]):
                w.writerow([i] + arr[i].tolist())

    write_emb_csv(emb_lowdim_path, emb_lowdim)

    # ----- Summary by epoch (from validation callback) -----
    if len(val_cb.per_epoch) > 0:
        print("\n[SUMMARY] Val exact/micro by epoch")
        print("  Epoch  |  Exact Match  |  Micro-F1")
        print("  ------ | ------------- | ---------")
        for ep, ex, mi in val_cb.per_epoch:
            print(f"  {ep:>5.2f}  |    {ex:>8.4f}   |  {mi:>8.4f}")

    print(f"\n[FINAL] exact_match(thr*)={exact_match:.4f} | micro_f1@0.5={micro_f1_05:.4f} | macro_f1@0.5={macro_f1_05:.4f} | "
          f"avg_kappa@0.5={avg_kappa:.4f} | best_epoch={val_cb.best_epoch} | best_val_exact={val_cb.best_exact:.4f}")
    print(f"[SAVE]  Wrote TRAIN per-sample probabilities: {probs_path}")
    print(f"[SAVE]  Wrote TRAIN low-dim embeddings:      {emb_lowdim_path}")


if __name__ == "__main__":
    main()
