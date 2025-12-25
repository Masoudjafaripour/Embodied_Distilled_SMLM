import os
import json
from dataclasses import dataclass
from typing import Dict, List

import torch
from PIL import Image
from datasets import load_dataset

from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    TrainingArguments,
    Trainer,
)

from peft import LoraConfig, get_peft_model

# =========================
# Config
# =========================
MODEL_ID = os.environ.get("MODEL_ID", None)
DATA_PATH = os.environ.get("DATA_PATH", "data/train.jsonl")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs/student")

MAX_LENGTH = 256

SYSTEM_PROMPT = (
    "You are a robot with a top-down view.\n"
    "Answer the spatial question and produce a collision-free action plan.\n"
    "Use only actions: UP DOWN LEFT RIGHT.\n"
)

# =========================
# Dataset loading
# =========================
def load_jsonl(path):
    return load_dataset("json", data_files=path, split="train")

def build_input(question: str) -> str:
    return f"{SYSTEM_PROMPT}\nQuestion: {question}\n"

def build_target(answer: str, reasoning: str, actions: str) -> str:
    return (
        f"Answer: {answer}\n"
        f"Reasoning: {reasoning}\n"
        f"Actions: {actions}"
    )

# =========================
# Collator
# =========================
@dataclass
class DistillCollator:
    processor: any
    max_length: int

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        images = [Image.open(ex["image"]).convert("RGB") for ex in batch]

        inputs = [
            build_input(ex["question"])
            for ex in batch
        ]

        targets = [
            build_target(
                ex["answer"],
                ex["teacher_reasoning"],
                ex["teacher_actions"],
            )
            for ex in batch
        ]

        model_inputs = self.processor(
            images=images,
            text=inputs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        with self.processor.as_target_processor():
            labels = self.processor(
                text=targets,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )["input_ids"]

        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels
        return model_inputs

# =========================
# Main training
# =========================
def main():
    if MODEL_ID is None:
        raise ValueError("Set MODEL_ID env variable to a small Vision2Seq model.")

    print(f"Loading dataset from {DATA_PATH}")
    dataset = load_jsonl(DATA_PATH)

    print(f"Loading model: {MODEL_ID}")
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # =========================
    # LoRA (lightweight finetuning)
    # =========================
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # =========================
    # Training setup
    # =========================
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=1,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        logging_steps=25,
        save_steps=500,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        bf16=torch.cuda.is_available(),
        report_to="none",
        remove_unused_columns=False,
    )

    collator = DistillCollator(
        processor=processor,
        max_length=MAX_LENGTH,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    print("Starting distillation training...")
    trainer.train()

    print("Saving model...")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

    print("Training complete.")
    print(f"Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
