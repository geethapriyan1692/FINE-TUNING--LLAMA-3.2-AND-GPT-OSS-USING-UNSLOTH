# gguf_export.py
from transformers import AutoTokenizer
from unsloth import FastLanguageModel

SRC = "outputs_gptoss_20b_bf16_lora-merged-16bit"   # merged 16-bit folder
OUT = "gguf_out"                                    # will be created

tok = AutoTokenizer.from_pretrained(SRC, use_fast=True)
# Load once via Unsloth if not already in memory (safe even after training):
model, _ = FastLanguageModel.from_pretrained(
    model_name=SRC, dtype="bfloat16", load_in_4bit=False, full_finetuning=False
)

# Pick a quant: "f16" (largest), "q8_0", "q6_k", "q5_k_m", "q4_k_m" (popular)
model.save_pretrained_gguf(OUT, tok, quantization_method="q4_k_m")
print("GGUF written to:", OUT)
