# dynamic_pipeline.py
"""
Dynamic fine-tuning pipeline (replacement for static_pipeline.py).
- Uses a hybrid technique: heuristic baseline (based on dataset size + model size) + Optuna to refine.
- Optimized for small datasets (100 - 3000 samples) and multicard (2x RTX A6000 recommended).
- Drop-in replacement for your static script: keep your DATA_DIR/Modelfile conventions.

Notes:
- This script keeps the original merge / gguf / ollama steps intact.
- You should install optuna in the environment: `pip install optuna`.
- Optuna trials will run training & evaluation; reduce n_trials if compute is limited.

"""

import os
import subprocess
import shutil
from pathlib import Path
import math
import tempfile
import json

import torch
import optuna
from datasets import load_dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from transformers import set_seed
from trl import SFTTrainer, SFTConfig
from tokenizers import Tokenizer
import sentencepiece as spm

# --------------------
# Config (user-editable defaults)
# --------------------
DATA_DIR = "./dataset_out/converted"
OUT_DIR = "outputs_gptoss_20b_bf16_lora"
BASE = "unsloth/gpt-oss-20b-BF16"
SEED = 3407
MAX_SEQ_LEN = 4096
DO_MERGE = True
CUSTOM_MODEFILE = "custom_Modelfile.txt"
LLAMACPP_DIR = "llama.cpp"
QUANTS = ["f16", "q8_0"]

# Optuna / search config
DEFAULT_N_TRIALS = 12
OPTUNA_TIMEOUT = None  # seconds or None

# --------------------
# Helpers (unchanged / lightly modified from your original)
# --------------------
SCRIPT_DIR = Path(__file__).resolve().parent


def load_split(name: str):
    path = Path(DATA_DIR) / f"{name}.jsonl"
    if not path.exists():
        return None
    return load_dataset("json", data_files=str(path), split="train")


# (keep write_modelfile/find_llama helpers from original script)
# ... for brevity I'm reusing your original implementations exactly as-is.

def write_modelfile(custom_modelfile: Path, chosen_gguf: str, out_path: Path):
    if custom_modelfile.exists():
        try:
            lines = custom_modelfile.read_bytes().decode("utf-8", errors="ignore").splitlines()
        except Exception as e:
            print(f"[WARN] Could not read {custom_modelfile} cleanly: {e}, falling back to default.")
            lines = []

        new_lines = []
        for line in lines:
            if line.strip().startswith("FROM "):
                new_lines.append(f"FROM {chosen_gguf}")
            else:
                new_lines.append(line)

        if not new_lines:
            new_lines = [f"FROM {chosen_gguf}", 'TEMPLATE "{{ .Prompt }}"']

        out_path.write_text("\n".join(new_lines), encoding="utf-8")
        print(f"[INFO] Using custom Modelfile with updated FROM: {chosen_gguf}")
    else:
        out_path.write_text(
            f"FROM {chosen_gguf}\nTEMPLATE \"{{{{ .Prompt }}}}\"",
            encoding="utf-8"
        )
        print(f"[INFO] Default Modelfile written (FROM {chosen_gguf})")


def find_convert_script_recursive(llamacpp_dir: Path):
    if not llamacpp_dir.exists():
        return None
    candidates = [p for p in llamacpp_dir.rglob("*.py") if ("convert" in p.name.lower() and "gguf" in p.name.lower())]
    if candidates:
        candidates_sorted = sorted(candidates, key=lambda p: (len(p.parts), p.name))
        chosen = candidates_sorted[0]
        print(f"[DEBUG] Found convert script: {chosen}")
        return chosen
    return None


def find_llama_quantize_binary(llamacpp_dir: Path):
    if not llamacpp_dir.exists():
        return None
    for name in ("llama-quantize", "llama"):
        found = list(llamacpp_dir.rglob(name))
        if found:
            print(f"[DEBUG] Found build binary: {found[0]}")
            return found[0]
    maybe = llamacpp_dir / "build" / "bin" / "llama-quantize"
    if maybe.exists():
        print(f"[DEBUG] Found build binary at: {maybe}")
        return maybe
    return None


def ensure_llama_cpp(llamacpp_dir: Path):
    llamacpp_dir = llamacpp_dir.resolve()
    print(f"[INFO] ensure_llama_cpp check path: {llamacpp_dir}")

    convert_script = find_convert_script_recursive(llamacpp_dir)
    build_binary = find_llama_quantize_binary(llamacpp_dir)

    if not llamacpp_dir.exists():
        print(f"[INFO] llama.cpp folder not found at {llamacpp_dir}. Cloning...")
        subprocess.run(["git", "clone", "https://github.com/ggerganov/llama.cpp.git", str(llamacpp_dir)], check=True)
        convert_script = find_convert_script_recursive(llamacpp_dir)
        build_binary = find_llama_quantize_binary(llamacpp_dir)
    else:
        if convert_script:
            print(f"[INFO] Found convert script at {convert_script}. Skipping git pull.")
        else:
            git_dir = llamacpp_dir / ".git"
            if git_dir.exists():
                print(f"[INFO] llama.cpp exists but no convert script found; running git -C pull to update repo...")
                subprocess.run(["git", "-C", str(llamacpp_dir), "pull"], check=True)
                convert_script = find_convert_script_recursive(llamacpp_dir)
                build_binary = find_llama_quantize_binary(llamacpp_dir)
            else:
                print(f"[WARN] {llamacpp_dir} exists but does not look like a git repo and has no convert script.")
                raise FileNotFoundError(f"llama.cpp directory exists at {llamacpp_dir} but appears incomplete (no convert script, not a git repo). Please fix or remove the directory.")

    build_binary = build_binary or find_llama_quantize_binary(llamacpp_dir)
    if build_binary:
        print("[INFO] Build binary already present; skipping build step.")
        return

    build_dir = llamacpp_dir / "build"
    build_dir.mkdir(exist_ok=True)
    print(f"[INFO] Building llama.cpp in {build_dir}...")
    subprocess.run(["cmake", ".."], cwd=build_dir, check=True)
    subprocess.run(["cmake", "--build", ".", "-j"], cwd=build_dir, check=True)
    print("[INFO] llama.cpp build complete ")


def rebuild_tokenizer_model_from_json(tokenizer_json_path: Path, output_path: Path):
    print(f"[INFO] Rebuilding tokenizer.model from {tokenizer_json_path}")

    tok = Tokenizer.from_file(str(tokenizer_json_path))
    vocab = tok.get_vocab()

    vocab_txt = output_path.with_suffix(".vocab.txt")
    with open(vocab_txt, "w", encoding="utf-8") as f:
        for token, idx in vocab.items():
            f.write(f"{token} {idx}\n")

    spm.SentencePieceTrainer.Train(
        f"--input={vocab_txt} "
        f"--model_prefix={output_path.with_suffix('')} "
        f"--vocab_size={len(vocab)} "
        "--character_coverage=1.0 "
        "--model_type=bpe"
    )
    print(f"[INFO] Rebuilt tokenizer.model at {output_path}")


def ensure_tokenizer_model(tokenizer, merged_dir: Path, base_model_id: str):
    tok_path = merged_dir / "tokenizer.model"
    tokenizer.save_pretrained(merged_dir)
    if not tok_path.exists() or tok_path.stat().st_size == 0:
        print("[WARN] tokenizer.model missing or empty, attempting rebuild...")
        tok_json = merged_dir / "tokenizer.json"
        if tok_json.exists():
            rebuild_tokenizer_model_from_json(tok_json, tok_path)

    if not tok_path.exists() or tok_path.stat().st_size == 0:
        raise FileNotFoundError(
            "tokenizer.model is still missing/empty! "
            f"Tried saving and rebuilding from tokenizer.json ({base_model_id})."
        )

# --------------------
# Dynamic heuristics
# --------------------

def get_device_info():
    # Force using available GPUs; user said they have 2x A6000
    ngpus = torch.cuda.device_count()
    if ngpus == 0:
        print("[WARN] No GPUs detected; falling back to CPU (very slow).")
    else:
        print(f"[INFO] Detected {ngpus} CUDA device(s)")
    return ngpus


def get_dynamic_params(dataset_size: int, ngpus: int):
    """Return initial heuristic params based on dataset size and number of GPUs.
    Suitable defaults for small datasets (100-3000 samples).
    """
    # Epochs heuristic: smaller datasets often need more epochs, but not too many
    if dataset_size < 200:
        epochs = 12
    elif dataset_size < 500:
        epochs = 10
    elif dataset_size < 1500:
        epochs = 8
    else:
        epochs = 6

    # target effective batch size (across all GPUs)
    # aim for 32 for stability, but clamp for very small datasets
    target_eff_batch = min(64, max(8, dataset_size // 10))
    target_eff_batch = max(8, min(target_eff_batch, 64))

    # per-device batch size (minimum 1)
    per_device_bs = max(1, target_eff_batch // max(1, ngpus))

    # gradient accumulation steps to achieve target_eff_batch exactly (if needed)
    grad_accum = max(1, target_eff_batch // (per_device_bs * max(1, ngpus)))

    # compute max_steps conservatively (total updates = (dataset_size/eff_batch)*epochs)
    steps_per_epoch = math.ceil(dataset_size / (per_device_bs * grad_accum * max(1, ngpus)))
    max_steps = steps_per_epoch * epochs

    # learning rate: base scaled by effective batch size
    base_lr = 5e-5
    lr = base_lr * ( (per_device_bs * grad_accum * max(1, ngpus)) / 32.0 )
    lr = max(1e-6, min(lr, 5e-4))

    # warmup steps as fraction of max steps (small datasets benefit from small warmup)
    warmup_steps = max(10, int(0.03 * max_steps))

    return {
        "epochs": epochs,
        "per_device_bs": per_device_bs,
        "grad_accum": grad_accum,
        "max_steps": max_steps,
        "learning_rate": lr,
        "warmup_steps": warmup_steps,
        "target_eff_batch": per_device_bs * grad_accum * max(1, ngpus),
    }

# --------------------
# Optuna integration
# --------------------

def optuna_objective(trial: optuna.trial.Trial, dataset_size: int, train_ds, val_ds, model_template):
    ngpus = get_device_info()
    baseline = get_dynamic_params(dataset_size, ngpus)

    # Suggest around baseline
    lr = trial.suggest_float("learning_rate", baseline["learning_rate"] / 3.0, baseline["learning_rate"] * 3.0, log=True)

    # try a few discrete choices for per-device batch size (power of 2 favored)
    max_per_device = min(8, max(1, baseline["per_device_bs"] * 2))
    per_dev_options = [1, 2, 4, 8]
    per_device_bs = trial.suggest_categorical("per_device_bs", [p for p in per_dev_options if p <= max_per_device])

    grad_accum = trial.suggest_int("grad_accum", 1, max(8, baseline["grad_accum"]))
    epochs = trial.suggest_int("epochs", max(1, baseline["epochs"] - 3), baseline["epochs"] + 2)
    lora_dropout = trial.suggest_float("lora_dropout", 0.0, 0.3)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)

    # recompute max_steps and warmup with chosen settings
    ngpus = max(1, ngpus)
    eff_batch = per_device_bs * grad_accum * ngpus
    steps_per_epoch = math.ceil(dataset_size / eff_batch)
    max_steps = steps_per_epoch * epochs
    warmup_steps = max(5, int(0.03 * max_steps))

    print(f"[TRIAL] lr={lr:.2e} per_device_bs={per_device_bs} grad_accum={grad_accum} epochs={epochs} lora_dropout={lora_dropout:.3f} eff_batch={eff_batch} max_steps={max_steps}")

    # Instantiate model (fresh for each trial) from template
    model, tokenizer = model_template()
    model = FastLanguageModel.for_training(model)

    # apply LoRA (small r to reduce memory/compute for trials)
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_alpha=32,
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )

    # Tokenizer already prepared by model_template

    # Prepare SFTTrainer
    args = SFTConfig(
        output_dir=str(Path(OUT_DIR) / f"optuna_trial_{trial.number}"),
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        max_steps=max_steps,
        learning_rate=lr,
        warmup_steps=warmup_steps,
        logging_steps=max(1, min(50, steps_per_epoch)),
        eval_strategy="steps" if val_ds is not None else "no",
        eval_steps=max(5, steps_per_epoch // 2) if val_ds is not None else None,
        weight_decay=weight_decay,
        lr_scheduler_type="linear",
        optim="adamw_torch",
        seed=SEED,
        report_to="none",
        bf16=True,
        packing=False,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=args,
    )

    # Run training. We rely on the trainer.eval after training to get a validation loss/metric.
    try:
        trainer.train(resume_from_checkpoint=False)
    except Exception as e:
        print(f"[TRIAL ERROR] training failed: {e}")
        # tell optuna this trial failed
        raise

    # Evaluate on validation set (if present) to get objective
    if val_ds is not None:
        eval_metrics = trainer.evaluate()
        # look for key like 'eval_loss' in returned metrics
        val_loss = None
        for k, v in eval_metrics.items():
            if k.lower().endswith("loss"):
                val_loss = float(v)
                break
        if val_loss is None:
            # fallback to any metric
            val_loss = float(list(eval_metrics.values())[0])
    else:
        # If no val set, use train metrics (not ideal)
        train_metrics = trainer.evaluate(train_ds)
        val_loss = float(list(train_metrics.values())[0])

    # cleanup trainer artifacts to save disk (optional)
    # shutil.rmtree(args.output_dir, ignore_errors=True)

    # report to optuna
    trial.report(val_loss, step=0)
    return val_loss


# --------------------
# Model template factory
# --------------------

def make_model_template():
    """Return a callable that loads the base model + tokenizer and applies chat template. This avoids reusing state across trials."""
    def loader():
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=BASE,
            max_seq_length=MAX_SEQ_LEN,
            dtype=torch.bfloat16,
            load_in_4bit=False,
            full_finetuning=False,
            device_map="auto",
        )
        model = FastLanguageModel.for_training(model)
        tokenizer = get_chat_template(tokenizer, chat_template="unsloth")
        if tokenizer.eos_token is None:
            tokenizer.eos_token = "</s>"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer
    return loader


# --------------------
# Main dynamic pipeline
# --------------------

def main(n_trials: int = DEFAULT_N_TRIALS):
    set_seed(SEED)

    train_ds = load_split("train")
    val_ds = load_split("val")

    if train_ds is None:
        raise FileNotFoundError(f"Missing {DATA_DIR}/train.jsonl")
    assert "messages" in train_ds.column_names, "Dataset must have 'messages' field!"

    # convert messages -> text (same as original)
    def to_text(batch):
        texts = [
            get_chat_template(None, chat_template="unsloth").apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            if False else None
        ]
        # The above is placeholder; we will use the same tokenizer pipeline below after loading model.
        return {"text": ["" for _ in batch["messages"]]}

    # figure dataset size
    dataset_size = len(train_ds)
    print(f"[INFO] Train size: {dataset_size}")

    # get device info
    ngpus = get_device_info()

    # baseline heuristics
    baseline = get_dynamic_params(dataset_size, ngpus)
    print(f"[INFO] Baseline heuristics: {json.dumps(baseline, indent=2)}")

    # load base model once to prepare tokenizer and text mapping
    model0, tokenizer0 = FastLanguageModel.from_pretrained(
        model_name=BASE,
        max_seq_length=MAX_SEQ_LEN,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        full_finetuning=False,
        device_map="auto",
    )
    model0 = FastLanguageModel.for_training(model0)
    tokenizer0 = get_chat_template(tokenizer0, chat_template="unsloth")
    if tokenizer0.eos_token is None:
        tokenizer0.eos_token = "</s>"
    if tokenizer0.pad_token is None:
        tokenizer0.pad_token = tokenizer0.eos_token

    # prepare mapping using tokenizer0
    def to_text_with_tokenizer(batch):
        texts = [
            tokenizer0.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            for msgs in batch["messages"]
        ]
        return {"text": texts}

    keep = ["messages"]
    train_ds = train_ds.map(lambda b: to_text_with_tokenizer(b), batched=True, remove_columns=[c for c in train_ds.column_names if c not in keep])
    if val_ds is not None:
        val_ds = val_ds.map(lambda b: to_text_with_tokenizer(b), batched=True, remove_columns=[c for c in val_ds.column_names if c not in keep])

    # Optuna search
    model_loader = make_model_template()

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner())
    print(f"[INFO] Starting Optuna search: n_trials={n_trials}")
    study.optimize(lambda trial: optuna_objective(trial, dataset_size, train_ds, val_ds, model_loader), n_trials=n_trials, timeout=OPTUNA_TIMEOUT)

    print("[INFO] Optuna search finished. Best params:")
    print(study.best_params)

    # Final training with best params
    best = study.best_params

    # ensure we have all keys with fallbacks to baseline
    per_device_bs = int(best.get("per_device_bs", baseline["per_device_bs"]))
    grad_accum = int(best.get("grad_accum", baseline["grad_accum"]))
    epochs = int(best.get("epochs", baseline["epochs"]))
    lr = float(best.get("learning_rate", baseline["learning_rate"]))
    lora_dropout = float(best.get("lora_dropout", 0.05)) if "lora_dropout" in best else 0.05
    weight_decay = float(best.get("weight_decay", 0.01)) if "weight_decay" in best else 0.01

    steps_per_epoch = math.ceil(dataset_size / (per_device_bs * grad_accum * max(1, ngpus)))
    max_steps = steps_per_epoch * epochs
    warmup_steps = max(10, int(0.03 * max_steps))

    print(f"[FINAL] per_device_bs={per_device_bs} grad_accum={grad_accum} epochs={epochs} lr={lr:.2e} max_steps={max_steps} warmup={warmup_steps} lora_dropout={lora_dropout}")

    # Recreate a final model + apply LoRA (bigger r from original)
    model_final, tokenizer_final = FastLanguageModel.from_pretrained(
        model_name=BASE,
        max_seq_length=MAX_SEQ_LEN,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        full_finetuning=False,
        device_map="auto",
    )
    model_final = FastLanguageModel.for_training(model_final)
    tokenizer_final = get_chat_template(tokenizer_final, chat_template="unsloth")
    if tokenizer_final.eos_token is None:
        tokenizer_final.eos_token = "</s>"
    if tokenizer_final.pad_token is None:
        tokenizer_final.pad_token = tokenizer_final.eos_token

    model_final = FastLanguageModel.get_peft_model(
        model_final,
        r=64,
        target_modules=["q_proj", "v_proj"],
        lora_alpha=32,
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )

    final_args = SFTConfig(
        output_dir=OUT_DIR,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        max_steps=max_steps,
        learning_rate=lr,
        warmup_steps=warmup_steps,
        logging_steps=max(1, min(50, steps_per_epoch)),
        eval_strategy="steps" if val_ds is not None else "no",
        eval_steps=max(50, steps_per_epoch // 2) if val_ds is not None else None,
        weight_decay=weight_decay,
        lr_scheduler_type="linear",
        optim="adamw_torch",
        seed=SEED,
        report_to="none",
        bf16=True,
        packing=False,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model_final,
        tokenizer=tokenizer_final,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=final_args,
    )

    trainer.train(resume_from_checkpoint=False)

    # Save outputs
    os.makedirs(OUT_DIR, exist_ok=True)
    model_final.save_pretrained(OUT_DIR)
    tokenizer_final.save_pretrained(OUT_DIR)
    print(f"[OK] LoRA saved: {OUT_DIR}")

    # Merge + GGUF + Ollama (same as original)
    if DO_MERGE:
        merged_dir = OUT_DIR + "-merged"
        if Path(merged_dir).exists():
            shutil.rmtree(merged_dir)

        print(f"[INFO] Merging adapters into: {merged_dir}")
        if hasattr(FastLanguageModel, "merge_lora"):
            FastLanguageModel.merge_lora(
                model_final,
                lora_model_dir=OUT_DIR,
                save_dir=merged_dir,
                dtype=torch.bfloat16,
            )
        else:
            model_final.save_pretrained_merged(
                merged_dir, tokenizer_final, save_method="merged_16bit",
            )

        ensure_tokenizer_model(tokenizer_final, Path(merged_dir), BASE)
        ensure_llama_cpp(Path(LLAMACPP_DIR))

        convert_script = find_convert_script_recursive(Path(LLAMACPP_DIR))
        if not convert_script:
            raise FileNotFoundError("Could not find convert_hf_to_gguf.py or similar inside llama.cpp after install/build.")

        ggufs = []
        for outtype in QUANTS:
            gguf_out = f"{merged_dir}-{outtype}.gguf"
            cmd = [
                "python3", str(convert_script),
                merged_dir,
                "--outfile", gguf_out,
                "--outtype", outtype,
            ]
            print(f"[INFO] Converting to GGUF ({outtype}) -> {gguf_out}")
            subprocess.run(cmd, check=True)
            ggufs.append(gguf_out)

        chosen_gguf = ggufs[-1]
        write_modelfile(Path(CUSTOM_MODEFILE), chosen_gguf, Path("Modelfile"))
        model_name = Path(OUT_DIR).name
        subprocess.run(["ollama", "create", model_name, "-f", "Modelfile"], check=True)
        print(f"[OK] Ollama model created: {model_name}")

    print("[DONE] Dynamic pipeline complete.")


if __name__ == "__main__":
    main(n_trials=DEFAULT_N_TRIALS)
