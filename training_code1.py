#!/usr/bin/env python3
"""
train_gpt_oss_optuna.py

Dynamic finetuning pipeline:
 - argparse CLI
 - optional Optuna hyperparameter search (with SQLite storage)
 - dataset-aware steps computation (auto when --steps=0)
 - LoRA training via Unsloth + TRL
 - merge, multi-GGUF export (via llama.cpp), and Ollama model creation
"""

import argparse
import subprocess
import shutil
import time
import math
from pathlib import Path
import warnings

import optuna
from optuna.pruners import MedianPruner

import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from transformers import set_seed
from trl import SFTTrainer, SFTConfig

# --------------------
# Defaults
# --------------------
DEFAULT_BASE = "unsloth/gpt-oss-20b-BF16"
DEFAULT_SEED = 3407
DEFAULT_MAX_SEQ_LEN = 1024
DEFAULT_PER_DEVICE_BS = 1
DEFAULT_GRAD_ACCUM = 4
DEFAULT_MAX_STEPS = 0  # 0 => compute from dataset & epochs
DEFAULT_EPOCHS = 3
DEFAULT_LR = 1.5e-4
DEFAULT_OUT_DIR = "outputs_gptoss_lora"
DEFAULT_LLAMACPP_DIR = "./llama.cpp"
DEFAULT_OLLAMA_NAME = "gptoss-custom"
DEFAULT_CUSTOM_MODEFILE = "custom_Modelfile.txt"
DEFAULT_QUANTS = "f16,q4_k_m"
DEFAULT_OPTUNA_DB = "sqlite:///optuna_finetune.db"

# --------------------
# CLI
# --------------------
def parse_args():
    p = argparse.ArgumentParser(description="GPT-OSS dynamic finetuning pipeline")

    p.add_argument("--dataset", type=str, required=True, help="Dataset folder with train.jsonl (and val/test optional)")
    p.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    p.add_argument("--base", type=str, default=DEFAULT_BASE)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)

    # training params
    p.add_argument("--steps", type=int, default=DEFAULT_MAX_STEPS, help="Training steps (0 = auto compute from dataset)")
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Epochs if steps=0")
    p.add_argument("--per-device-batch-size", type=int, default=DEFAULT_PER_DEVICE_BS)
    p.add_argument("--grad-accum", type=int, default=DEFAULT_GRAD_ACCUM)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--lora-r", type=int, default=64)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--target-modules", type=str, default="q_proj,v_proj")

    # export / ollama
    p.add_argument("--merge", action="store_true", help="Merge LoRA and export GGUF + create Ollama model")
    p.add_argument("--llama-dir", type=str, default=DEFAULT_LLAMACPP_DIR)
    p.add_argument("--quants", type=str, default=DEFAULT_QUANTS)
    p.add_argument("--ollama-name", type=str, default=DEFAULT_OLLAMA_NAME)
    p.add_argument("--custom-modelfile", type=str, default=DEFAULT_CUSTOM_MODEFILE)
    p.add_argument("--no-ollama", action="store_true", help="Do not call ollama create (just write Modelfile)")

    # optuna
    p.add_argument("--optuna-trials", type=int, default=0, help="Number of Optuna trials (0=disabled)")
    p.add_argument("--optuna-storage", type=str, default=DEFAULT_OPTUNA_DB, help="Optuna storage URL (default sqlite file)")
    p.add_argument("--optuna-n-startup-trials", type=int, default=3)
    p.add_argument("--optuna-pruner-warmup-steps", type=int, default=8)

    # misc
    p.add_argument("--keep-tmp", action="store_true")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()

# --------------------
# Helpers
# --------------------
def load_split(dataset_dir: str, name: str):
    path = Path(dataset_dir) / f"{name}.jsonl"
    if not path.exists():
        return None
    return load_dataset("json", data_files=str(path), split="train")

def compute_steps(train_ds, bs, accum, epochs):
    n = len(train_ds)
    steps_per_epoch = math.ceil(n / (bs * accum))
    return steps_per_epoch * epochs

def find_convert_script(llama_dir: str):
    for c in ["convert-hf-to-gguf.py", "convert.py"]:
        p = Path(llama_dir) / c
        if p.exists():
            return p
    raise FileNotFoundError(f"No convert script found in {llama_dir}")

def run_convert(script: Path, hf_dir: str, out_gguf: str, q: str):
    cmd = ["python3", str(script), hf_dir, "--outfile", out_gguf, "--outtype", q]
    print("[INFO] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def write_modelfile(template: Path, gguf: str, out: Path):
    if template.exists():
        lines = template.read_text().splitlines()
        new_lines = []
        replaced = False
        for l in lines:
            if l.strip().upper().startswith("FROM "):
                new_lines.append(f"FROM {gguf}")
                replaced = True
            else:
                new_lines.append(l)
        if not replaced:
            new_lines = [f"FROM {gguf}", ""] + new_lines
        out.write_text("\n".join(new_lines))
    else:
        out.write_text(f"FROM {gguf}\nTEMPLATE \"\"\"\n{{{{ .Prompt }}}}\n\"\"\"\n")

# --------------------
# Training
# --------------------
def to_text_factory(tokenizer):
    def to_text(batch):
        texts = []
        if "messages" in batch:
            for conv in batch["messages"]:
                texts.append(tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False))
        elif "conversation" in batch:
            for conv in batch["conversation"]:
                texts.append(tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False))
        elif "question" in batch and "answer" in batch:
            for q, a in zip(batch["question"], batch["answer"]):
                conv = [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
                texts.append(tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False))
        else:
            raise ValueError("Dataset must have messages OR conversation OR question+answer")
        return {"text": texts}
    return to_text

def train_and_eval(args, lr=None, lora_r=None, lora_alpha=None, lora_dropout=None, steps_override=None, trial=None):
    set_seed(args.seed)

    train_ds = load_split(args.dataset, "train")
    val_ds = load_split(args.dataset, "val")

    if train_ds is None:
        raise FileNotFoundError("train.jsonl missing")

    steps = steps_override or (args.steps if args.steps > 0 else compute_steps(train_ds, args.per_device_batch_size, args.grad_accum, args.epochs))

    # model + tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base,
        max_seq_length=args.max_seq_len,
        dtype=torch.bfloat16,
        load_in_4bit=True,
        full_finetuning=False,
        device_map="auto",
    )
    model = FastLanguageModel.for_training(model)
    tokenizer = get_chat_template(tokenizer, chat_template="unsloth")

    train_ds = train_ds.map(to_text_factory(tokenizer), batched=True)
    if val_ds: val_ds = val_ds.map(to_text_factory(tokenizer), batched=True)

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r or args.lora_r,
        target_modules=[m.strip() for m in args.target_modules.split(",")],
        lora_alpha=lora_alpha or args.lora_alpha,
        lora_dropout=lora_dropout or args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=SFTConfig(
            output_dir=args.out_dir,
            per_device_train_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=args.grad_accum,
            max_steps=steps,
            learning_rate=lr or args.lr,
            logging_steps=10,
            eval_strategy="steps" if val_ds else "no",
            eval_steps=50 if val_ds else None,
            bf16=True,
            dataset_text_field="text",
            save_strategy="steps",
            save_steps=100,
            save_total_limit=3,
        ),
    )

    trainer.train(resume_from_checkpoint=False)
    metrics = trainer.evaluate() if val_ds else {}
    return metrics.get("eval_loss", float("inf"))

# --------------------
# Optuna
# --------------------
def run_optuna(args):
    sampler = optuna.samplers.TPESampler(n_startup_trials=args.optuna_n_startup_trials)
    pruner = MedianPruner(n_warmup_steps=args.optuna_pruner_warmup_steps)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner,
                                storage=args.optuna_storage, load_if_exists=True)

    def objective(trial):
        lr = trial.suggest_float("lr", 5e-5, 2e-4, log=True)
        lora_r = trial.suggest_categorical("lora_r", [32, 64, 128])
        lora_alpha = trial.suggest_int("lora_alpha", 16, 64, step=16)
        lora_dropout = trial.suggest_float("lora_dropout", 0.0, 0.2)
        steps = 100  # short trial run
        return train_and_eval(args, lr, lora_r, lora_alpha, lora_dropout, steps, trial)

    study.optimize(objective, n_trials=args.optuna_trials)
    print("Best params:", study.best_params)
    return study.best_params

# --------------------
# Merge + GGUF + Ollama
# --------------------
def merge_and_export(args):
    merged_dir = args.out_dir + "-merged"
    if Path(merged_dir).exists():
        shutil.rmtree(merged_dir)

    model, _ = FastLanguageModel.from_pretrained(args.out_dir)
    FastLanguageModel.merge_lora(model, lora_model_dir=args.out_dir, save_dir=merged_dir, dtype=torch.bfloat16)

    _, tokenizer = FastLanguageModel.from_pretrained(args.base)
    tokenizer.save_pretrained(merged_dir)

    script = find_convert_script(args.llama_dir)
    ggufs = []
    for q in args.quants.split(","):
        gguf = f"{merged_dir}-{q}.gguf"
        run_convert(script, merged_dir, gguf, q)
        ggufs.append(gguf)

    write_modelfile(Path(args.custom_modelfile), ggufs[-1], Path("Modelfile"))
    if not args.no_ollama:
        subprocess.run(["ollama", "create", args.ollama_name, "-f", "Modelfile"], check=True)

    print("Exported GGUFs:", ggufs)

# --------------------
# Main
# --------------------
def main():
    args = parse_args()

    if args.optuna_trials > 0:
        best = run_optuna(args)
        args.lr = best.get("lr", args.lr)
        args.lora_r = best.get("lora_r", args.lora_r)
        args.lora_alpha = best.get("lora_alpha", args.lora_alpha)
        args.lora_dropout = best.get("lora_dropout", args.lora_dropout)

    val_loss = train_and_eval(args)
    print("Final val_loss:", val_loss)

    if args.merge:
        merge_and_export(args)

if __name__ == "__main__":
    main()
