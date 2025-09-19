# static_pipeline.py
import os
import subprocess
import shutil
from pathlib import Path
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from transformers import set_seed
from trl import SFTTrainer, SFTConfig
from tokenizers import Tokenizer
import sentencepiece as spm

# --------------------
# Config
# --------------------
DATA_DIR = "./dataset_out"                            
OUT_DIR  = "outputs_gptoss_20b_bf16_lora"
BASE     = "unsloth/gpt-oss-20b-BF16"
SEED = 3407
MAX_SEQ_LEN = 4096
PER_DEVICE_BS = 1
GRAD_ACCUM = 8
MAX_STEPS = 2
LR = 1.5e-4
DO_MERGE = True

# Always resolve paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
LLAMACPP_DIR = SCRIPT_DIR / "llama.cpp"
QUANTS = ["f16", "q8_0"]     # supported types for latest llama.cpp
CUSTOM_MODEFILE = "custom_Modelfile.txt"

# --------------------
# Helpers
# --------------------
def load_split(name: str):
    path = Path(DATA_DIR) / f"{name}.jsonl"
    if not path.exists():
        return None
    return load_dataset("json", data_files=str(path), split="train")

def write_modelfile(custom_modelfile: Path, chosen_gguf: str, out_path: Path):
    """Write Modelfile using custom template if available"""
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
    """Search recursively for a convert-*gguf*.py file. Returns Path or None."""
    if not llamacpp_dir.exists():
        return None
    # Look for any python filename that contains both 'convert' and 'gguf' (case-insensitive)
    candidates = [p for p in llamacpp_dir.rglob("*.py") if ("convert" in p.name.lower() and "gguf" in p.name.lower())]
    if candidates:
        # return the first sensible one (prefer one in repo root)
        candidates_sorted = sorted(candidates, key=lambda p: (len(p.parts), p.name))  # prefer shallower paths
        chosen = candidates_sorted[0]
        print(f"[DEBUG] Found convert script: {chosen}")
        return chosen
    return None

def find_llama_quantize_binary(llamacpp_dir: Path):
    """Search recursively for a llama-quantize or llama-* binary presence. Returns Path or None."""
    if not llamacpp_dir.exists():
        return None
    # search for typical binary names
    for name in ("llama-quantize", "llama"):
        found = list(llamacpp_dir.rglob(name))
        if found:
            print(f"[DEBUG] Found build binary: {found[0]}")
            return found[0]
    # also check build/bin
    maybe = llamacpp_dir / "build" / "bin" / "llama-quantize"
    if maybe.exists():
        print(f"[DEBUG] Found build binary at: {maybe}")
        return maybe
    return None

def ensure_llama_cpp(llamacpp_dir: Path):
    """
    Clone and build llama.cpp if missing. Skip clone/pull/rebuild when the repo already contains
    a convert script and/or built binaries. Will only update (git pull) when repo exists but looks incomplete.
    """
    llamacpp_dir = llamacpp_dir.resolve()
    print(f"[INFO] ensure_llama_cpp check path: {llamacpp_dir}")

    convert_script = find_convert_script_recursive(llamacpp_dir)
    build_binary = find_llama_quantize_binary(llamacpp_dir)

    if not llamacpp_dir.exists():
        # Clone since folder missing
        print(f"[INFO] llama.cpp folder not found at {llamacpp_dir}. Cloning...")
        subprocess.run(["git", "clone", "https://github.com/ggerganov/llama.cpp.git", str(llamacpp_dir)], check=True)
        # after clone, try finding scripts again
        convert_script = find_convert_script_recursive(llamacpp_dir)
        build_binary = find_llama_quantize_binary(llamacpp_dir)
    else:
        # directory exists
        if convert_script:
            # convert script exists somewhere in repo -> treat repo as complete enough; skip pull
            print(f"[INFO] Found convert script at {convert_script}. Skipping git pull.")
        else:
            # repo exists but missing convert script -> attempt to update repo (git pull) once
            git_dir = llamacpp_dir / ".git"
            if git_dir.exists():
                print(f"[INFO] llama.cpp exists but no convert script found; running git -C pull to update repo...")
                subprocess.run(["git", "-C", str(llamacpp_dir), "pull"], check=True)
                # refresh find
                convert_script = find_convert_script_recursive(llamacpp_dir)
                build_binary = find_llama_quantize_binary(llamacpp_dir)
            else:
                # directory exists but not a git repo and missing script -> try to clone into a temp and move? Fallback: raise
                print(f"[WARN] {llamacpp_dir} exists but does not look like a git repo and has no convert script.")
                raise FileNotFoundError(f"llama.cpp directory exists at {llamacpp_dir} but appears incomplete (no convert script, not a git repo). Please fix or remove the directory.")

    # After clone/pull, decide whether to build.
    build_binary = build_binary or find_llama_quantize_binary(llamacpp_dir)
    if build_binary:
        print("[INFO] Build binary already present; skipping build step.")
        return  # nothing else to do

    # No build binary found -> build llama.cpp
    build_dir = llamacpp_dir / "build"
    build_dir.mkdir(exist_ok=True)
    print(f"[INFO] Building llama.cpp in {build_dir}...")
    subprocess.run(["cmake", ".."], cwd=build_dir, check=True)
    subprocess.run(["cmake", "--build", ".", "-j"], cwd=build_dir, check=True)
    print("[INFO] llama.cpp build complete ")

def rebuild_tokenizer_model_from_json(tokenizer_json_path: Path, output_path: Path):
    """Rebuild tokenizer.model from tokenizer.json"""
    print(f"[INFO] Rebuilding tokenizer.model from {tokenizer_json_path}")

    tok = Tokenizer.from_file(str(tokenizer_json_path))
    vocab = tok.get_vocab()

    # Write vocab to a temporary file
    vocab_txt = output_path.with_suffix(".vocab.txt")
    with open(vocab_txt, "w", encoding="utf-8") as f:
        for token, idx in vocab.items():
            f.write(f"{token} {idx}\n")

    # Train SentencePiece model
    spm.SentencePieceTrainer.Train(
        f"--input={vocab_txt} "
        f"--model_prefix={output_path.with_suffix('')} "
        f"--vocab_size={len(vocab)} "
        "--character_coverage=1.0 "
        "--model_type=bpe"
    )
    print(f"[INFO] Rebuilt tokenizer.model at {output_path}")

def ensure_tokenizer_model(tokenizer, merged_dir: Path, base_model_id: str):
    """Ensure tokenizer.model exists, otherwise rebuild from tokenizer.json"""
    tok_path = merged_dir / "tokenizer.model"

    # Step 1: Save normally
    tokenizer.save_pretrained(merged_dir)

    # Step 2: If missing or empty, rebuild from tokenizer.json
    if not tok_path.exists() or tok_path.stat().st_size == 0:
        print("[WARN] tokenizer.model missing or empty, attempting rebuild...")
        tok_json = merged_dir / "tokenizer.json"
        if tok_json.exists():
            rebuild_tokenizer_model_from_json(tok_json, tok_path)

    # Step 3: Final check
    if not tok_path.exists() or tok_path.stat().st_size == 0:
        raise FileNotFoundError(
            "tokenizer.model is still missing/empty! "
            f"Tried saving and rebuilding from tokenizer.json ({base_model_id})."
        )

# --------------------
# Main
# --------------------
def main():
    set_seed(SEED)

    # --------------------
    # Load datasets
    # --------------------
    train_ds = load_split("train")
    val_ds   = load_split("val")

    if train_ds is None:
        raise FileNotFoundError(f"Missing {DATA_DIR}/train.jsonl")
    assert "messages" in train_ds.column_names, "Dataset must have 'messages' field!"

    # --------------------
    # Load base model + tokenizer
    # --------------------
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

    # --------------------
    # Text mapping from `messages`
    # --------------------
    def to_text(batch):
        texts = [
            tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            ) for msgs in batch["messages"]
        ]
        return {"text": texts}

    keep = ["messages"]
    train_ds = train_ds.map(to_text, batched=True, remove_columns=[c for c in train_ds.column_names if c not in keep])
    if val_ds is not None:
        val_ds = val_ds.map(to_text, batched=True, remove_columns=[c for c in val_ds.column_names if c not in keep])

    # --------------------
    # Add LoRA
    # --------------------
    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        target_modules=["q_proj","v_proj"],
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )

    # --------------------
    # Trainer
    # --------------------
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=SFTConfig(
            output_dir=OUT_DIR,
            per_device_train_batch_size=PER_DEVICE_BS,
            gradient_accumulation_steps=GRAD_ACCUM,
            max_steps=MAX_STEPS,
            learning_rate=LR,
            warmup_steps=10,
            logging_steps=10,
            eval_strategy="steps" if val_ds is not None else "no",
            eval_steps=50,
            weight_decay=0.01,
            lr_scheduler_type="linear",
            optim="adamw_torch",
            seed=SEED,
            report_to="none",
            bf16=True,
            packing=False,
            dataset_text_field="text",
        ),
    )

    trainer.train(resume_from_checkpoint=False)

    # --------------------
    # Save LoRA adapter
    # --------------------
    os.makedirs(OUT_DIR, exist_ok=True)
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"[OK] LoRA saved: {OUT_DIR}")

    # --------------------
    # Merge + GGUF + Ollama
    # --------------------
    if DO_MERGE:
        merged_dir = OUT_DIR + "-merged"
        if Path(merged_dir).exists():
            shutil.rmtree(merged_dir)

        print(f"[INFO] Merging adapters into: {merged_dir}")
        if hasattr(FastLanguageModel, "merge_lora"):
            FastLanguageModel.merge_lora(
                model,
                lora_model_dir=OUT_DIR,
                save_dir=merged_dir,
                dtype=torch.bfloat16,
            )
        else:
            model.save_pretrained_merged(
                merged_dir, tokenizer, save_method="merged_16bit",
            )

        # Ensure tokenizer.model is present (save or rebuild)
        ensure_tokenizer_model(tokenizer, Path(merged_dir), BASE)

        # Ensure llama.cpp is installed + built (cached). This will skip clone/pull if already complete.
        ensure_llama_cpp(Path(LLAMACPP_DIR))

        # Find convert script (robust search)
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

        # Write Modelfile (use last quant or pick manually)
        chosen_gguf = ggufs[-1]
        write_modelfile(Path(CUSTOM_MODEFILE), chosen_gguf, Path("Modelfile"))

        # Create Ollama model
        model_name = Path(OUT_DIR).name
        subprocess.run(["ollama", "create", model_name, "-f", "Modelfile"], check=True)
        print(f"[OK] Ollama model created: {model_name}")

    print("[DONE] Static pipeline complete.")

if __name__ == "__main__":
    main()
