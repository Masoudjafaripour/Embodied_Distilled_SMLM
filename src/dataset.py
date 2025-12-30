"""
Dataset and collator for embodied distillation.

Loads:
- image
- spatial question
- teacher answer / reasoning / actions

Produces:
(image + prompt) -> target text
"""

from dataclasses import dataclass
from typing import Dict, List

from PIL import Image
from datasets import load_dataset
import torch

from src.prompts import student_prompt, format_target


# =========================
# Dataset loader
# =========================
def load_distillation_dataset(jsonl_path: str):
    """
    Loads a JSONL distillation dataset using HuggingFace Datasets.

    Each line must contain:
      - image
      - question
      - answer
      - teacher_reasoning
      - teacher_actions
    """
    return load_dataset("json", data_files=jsonl_path, split="train")


# =========================
# Data collator
# =========================
@dataclass
class DistillationCollator:
    """
    Collator that prepares (image + text) -> labels
    for Vision-to-Sequence models.
    """
    processor: any
    max_length: int = 256

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        # ---- Load images ----
        images = [
            Image.open(ex["image"]).convert("RGB")
            for ex in batch
        ]

        # ---- Build prompts ----
        inputs = [
            student_prompt(ex["question"])
            for ex in batch
        ]

        targets = [
            format_target(
                ex["answer"],
                ex["teacher_reasoning"],
                ex["teacher_actions"],
            )
            for ex in batch
        ]

        # ---- Tokenize inputs ----
        model_inputs = self.processor(
            images=images,
            text=inputs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # ---- Tokenize targets ----
        with self.processor.as_target_processor():
            labels = self.processor(
                text=targets,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )["input_ids"]

        # ---- Mask padding tokens ----
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels

        return model_inputs
