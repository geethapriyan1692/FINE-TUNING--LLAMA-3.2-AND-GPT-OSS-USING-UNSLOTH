# train_gpt_oss_bf16.py
import os, json
from pathlib import Path
import torch
from datasets import load_dataset, Dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from transformers import set_seed, GenerationConfig
from trl import SFTTrainer, SFTConfig

# ---- Config ----
DATA_DIR = "./dataset_out"   # must contain train.jsonl, val.jsonl (optional)
OUT_DIR = "outputs_gptoss_20b_bf16_lora"
SEED = 3407
MAX_SEQ_LEN = 2048
PER_DEVICE_BS = 1
GRAD_ACCUM = 4
MAX_STEPS = 100
LR = 2e-4
REASONING_EFFORT = "medium"
MERGE_16BIT = True

# --- Helpers ---
def read_dataset(path: Path) -> Dataset:
    if path.suffix.lower() == ".jsonl":
        return load_dataset("json", data_files=str(path), split="train")
    elif path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, dict): data = [data]
        return Dataset.from_list(data)
    raise ValueError("Dataset must be .json or .jsonl")

def main(train_mode=True):
    set_seed(SEED)

    # --- Load datasets ---
    train_path = Path(DATA_DIR) / "train.jsonl"
    val_path   = Path(DATA_DIR) / "val.jsonl"

    train_ds = read_dataset(train_path) if train_path.exists() else None
    val_ds   = read_dataset(val_path) if val_path.exists() else None

    # --- Load model ---
    ckpt_dir = OUT_DIR if train_mode else OUT_DIR + "-merged-16bit"
    print(f"[INFO] Loading model from: {ckpt_dir}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=ckpt_dir,
        max_seq_length=MAX_SEQ_LEN,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        full_finetuning=False,
        device_map="auto",
    )

    model.config.attn_implementation = "flash_attention_2"
    tokenizer = get_chat_template(tokenizer, chat_template="unsloth")

    # --- Preprocessing ---
    def to_text(batch):
        convos = batch["messages"]
        texts = [
            tokenizer.apply_chat_template(
                c, tokenize=False, add_generation_prompt=False,
                reasoning_effort=REASONING_EFFORT
            ) for c in convos
        ]
        return {"text": texts}

    def preprocess(ds):
        return ds.map(
            to_text,
            batched=True,
            remove_columns=[c for c in ds.column_names if c != "messages"]
        )

    if train_ds is not None: train_ds = preprocess(train_ds)
    if val_ds   is not None: val_ds   = preprocess(val_ds)

    if train_mode:
        print(f"[INFO] Train={len(train_ds)} | Val={len(val_ds) if val_ds else 0}")
        print("[INFO] Sample processed text:\n", train_ds[0]["text"][:400])

        # --- Add LoRA ---
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules="all-linear",
            lora_alpha=16,
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=SEED,
        )

        # --- Trainer ---
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
                warmup_steps=5,
                logging_steps=10,
                eval_strategy="steps" if val_ds is not None else "no",
                eval_steps=50,
                weight_decay=0.01,
                lr_scheduler_type="linear",
                optim="adamw_torch",
                seed=SEED,
                report_to="none",
                bf16=True,
                packing=True,
                dataset_text_field="text",
                remove_unused_columns=False,
            ),
        )

        trainer.train(resume_from_checkpoint=True)

        # --- Save model ---
        os.makedirs(OUT_DIR, exist_ok=True)
        model.save_pretrained(OUT_DIR)
        tokenizer.save_pretrained(OUT_DIR)
        print(f"[OK] LoRA adapters saved to: {OUT_DIR}")

        if MERGE_16BIT:
            merged_dir = OUT_DIR + "-merged-16bit"
            model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
            print(f"[OK] Merged 16-bit model saved to: {merged_dir}")

    else:
        # --- Inference only (using merged model) ---
        print("[INFO] Running inference with merged model...")
        FastLanguageModel.for_inference(model)

        SYSTEM = "You are a highly knowledgeable assistant specialized in domain-specific instructions."
        USER   = "How to View the Dashboard?"

        if tokenizer.eos_token_id is None and tokenizer.eos_token is None:
            tokenizer.add_special_tokens({"eos_token": "<|end|>"})
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER},
        ]

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, reasoning_effort=REASONING_EFFORT
        )
        enc = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, return_attention_mask=True).to(model.device)

        gc = GenerationConfig(max_new_tokens=256, do_sample=False, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
        out = model.generate(**enc, generation_config=gc)
        print("[OUTPUT SAMPLE]", tokenizer.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    # ?? Set train_mode=False if you just want to run inference on merged model
    main()
