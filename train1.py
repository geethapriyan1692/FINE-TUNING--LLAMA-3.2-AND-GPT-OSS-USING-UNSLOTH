# train_gpt_oss_bf16_minimal.py
import os, json
from pathlib import Path
import torch
from datasets import load_dataset, Dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from transformers import set_seed, TextStreamer
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel

# ---- Config (hard-coded) ----
DATA = "normalized_messages.json"   # or .jsonl
OUT_DIR = "outputs_gptoss_20b_bf16_lora"
SEED = 3407
MAX_SEQ_LEN = 2048
PER_DEVICE_BS = 1
GRAD_ACCUM = 4
#EPOCHS = 10
MAX_STEPS = 100     # e.g., 60 for smoke test
LR = 2e-4
REASONING_EFFORT = "medium"
MERGE_16BIT = True   

def read_dataset(path: Path) -> Dataset:
    if path.suffix.lower() == ".jsonl":
        return load_dataset("json", data_files=str(path), split="train")
    elif path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, dict): data = [data]
        return Dataset.from_list(data)
    raise ValueError("Dataset must be .json or .jsonl")

def main():
    set_seed(SEED)
    ds = read_dataset(Path(DATA))
    assert "messages" in ds.column_names, "dataset rows must have 'messages'"

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/gpt-oss-20b-BF16",
        max_seq_length=MAX_SEQ_LEN,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        full_finetuning=False,
        device_map="cuda",
    )

    # Tell Transformers/Flash-attn which kernels to use
    model.config.attn_implementation = "flash_attention_2"    

    tokenizer = get_chat_template(tokenizer, chat_template="unsloth")

    def to_text(batch):
        convos = batch["messages"]
        texts = [
            tokenizer.apply_chat_template(
                c, tokenize=False, add_generation_prompt=False,
                reasoning_effort=REASONING_EFFORT
            ) for c in convos
        ]
        return {"text": texts}

    ds = ds.map(to_text, batched=True, remove_columns=[c for c in ds.column_names if c != "text"])
                                                                  


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

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        args=SFTConfig(
            output_dir=OUT_DIR,
            per_device_train_batch_size=PER_DEVICE_BS,
            gradient_accumulation_steps=GRAD_ACCUM,
            #num_train_epochs=None if MAX_STEPS else EPOCHS,
            max_steps=MAX_STEPS,
            learning_rate=LR,
            warmup_steps=5,
            logging_steps=10,
            weight_decay=0.01,
            lr_scheduler_type="linear",
            optim="adamw_torch",
            seed=SEED,
            report_to="none",
            bf16=True,
            packing=True,
            dataset_text_field="text",
        ),
    )
    trainer.train()
    #trainer.train(resume_from_checkpoint=True)


    os.makedirs(OUT_DIR, exist_ok=True)
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"[OK] LoRA adapters saved to: {OUT_DIR}")

    if MERGE_16BIT:
        merged_dir = OUT_DIR + "-merged-16bit"
        model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
        print(f"[OK] Merged 16-bit model saved to: {merged_dir}")

    # --- Sanity inference with attention mask ---
    
    from transformers import GenerationConfig

    SYSTEM = "You are a highly knowledgeable assistant specialized in domain-specific instructions. Always provide step-by-step, accurate, and context-aware answers. Never assume, always rely on the given information. Avoid general knowledge unless explicitly asked. Use concise and formal language."
    USER = "How to View the Dashboard?"

    FastLanguageModel.for_inference(model)

    if tokenizer.eos_token_id is None and tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "<|end|>"})
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    messages = [
        {"role":"system","content": SYSTEM},
        {"role":"user","content": USER},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        reasoning_effort=REASONING_EFFORT,
    )

    enc = tokenizer(prompt, return_tensors="pt",
                    padding=True, truncation=True,
                    return_attention_mask=True).to(model.device)

    gc = GenerationConfig(
        max_new_tokens=256,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    out = model.generate(**enc, generation_config=gc)
    print(tokenizer.decode(out[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
