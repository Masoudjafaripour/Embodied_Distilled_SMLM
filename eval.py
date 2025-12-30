import os
import json
import re
from typing import Tuple, List

import numpy as np
from PIL import Image

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from tqdm import tqdm

# =========================
# Config
# =========================
MODEL_DIR = os.environ.get("MODEL_DIR", "outputs/student")
DATA_PATH = os.environ.get("DATA_PATH", "data/train.jsonl")

GRID_SIZE = 16
CELL_SIZE = 20

ACTIONS = {"UP", "DOWN", "LEFT", "RIGHT"}
MOVE = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
}

MAX_NEW_TOKENS = 128

# =========================
# Parsing helpers
# =========================
def extract_answer(text: str) -> str:
    m = re.search(r"Answer:\s*(Yes|No)", text, re.IGNORECASE)
    return m.group(1).capitalize() if m else "Invalid"

def extract_actions(text: str) -> List[str]:
    return re.findall(r"\b(UP|DOWN|LEFT|RIGHT)\b", text.upper())

# =========================
# Image → grid decoding
# =========================
def decode_grid_from_image(img: Image.Image) -> Tuple[np.ndarray, Tuple[int,int], Tuple[int,int]]:
    """
    Decode obstacles, start, and goal from the rendered image.
    Assumes:
      - obstacles ~ dark gray
      - start ~ green
      - goal ~ red
    """
    img = img.resize((GRID_SIZE*CELL_SIZE, GRID_SIZE*CELL_SIZE))
    arr = np.array(img)

    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    start = goal = None

    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            patch = arr[
                y*CELL_SIZE:(y+1)*CELL_SIZE,
                x*CELL_SIZE:(x+1)*CELL_SIZE
            ]
            mean = patch.mean(axis=(0,1))

            if mean[0] < 80 and mean[1] < 80 and mean[2] < 80:
                grid[y,x] = 1  # obstacle
            elif mean[1] > 150 and mean[0] < 100:
                start = (x,y)
            elif mean[0] > 150 and mean[1] < 100:
                goal = (x,y)

    return grid, start, goal

# =========================
# Plan simulation
# =========================
def simulate_plan(grid, start, goal, actions: List[str]) -> bool:
    if start is None or goal is None:
        return False

    x, y = start
    H, W = grid.shape

    for a in actions:
        dx, dy = MOVE[a]
        nx, ny = x + dx, y + dy

        if not (0 <= nx < W and 0 <= ny < H):
            return False
        if grid[ny, nx] == 1:
            return False

        x, y = nx, ny

    return (x, y) == goal

# =========================
# Inference
# =========================
def infer(model, processor, image, question):
    prompt = (
        "You are a robot with a top-down view.\n"
        "Answer the spatial question and produce a collision-free action plan.\n"
        "Use only actions: UP DOWN LEFT RIGHT.\n\n"
        f"Question: {question}\n"
    )

    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt",
    ).to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    return processor.tokenizer.decode(output[0], skip_special_tokens=True)

# =========================
# Main evaluation
# =========================
def main(limit=None):
    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    total = 0
    answer_correct = 0
    valid_actions = 0
    feasible_plans = 0

    with open(DATA_PATH, "r") as f:
        lines = f.readlines()

    if limit:
        lines = lines[:limit]

    for line in tqdm(lines, desc="Evaluating"):
        ex = json.loads(line)
        img = Image.open(ex["image"]).convert("RGB")

        pred = infer(model, processor, img, ex["question"])

        pred_answer = extract_answer(pred)
        pred_actions = extract_actions(pred)

        # Metric 1: Answer accuracy
        if pred_answer == ex["answer"]:
            answer_correct += 1

        # Metric 2: Valid action syntax
        if len(pred_actions) > 0:
            valid_actions += 1

        # Metric 3: Feasible plan
        grid, start, goal = decode_grid_from_image(img)
        if pred_actions and simulate_plan(grid, start, goal, pred_actions):
            feasible_plans += 1

        total += 1

    print("\n===== Evaluation Results =====")
    print(f"Samples evaluated: {total}")
    print(f"Answer accuracy:     {answer_correct/total:.3f}")
    print(f"Valid action rate:   {valid_actions/total:.3f}")
    print(f"Feasible plan rate:  {feasible_plans/total:.3f}")

if __name__ == "__main__":
    main(limit=500)  # adjust or set None
