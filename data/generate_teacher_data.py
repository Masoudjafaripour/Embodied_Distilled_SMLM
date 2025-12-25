import os
import json
import time
import hashlib
import random
from typing import Dict, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

from openai import OpenAI

# =========================
# Config
# =========================
OUT_DIR = "data"
IMG_DIR = os.path.join(OUT_DIR, "images")
JSONL_PATH = os.path.join(OUT_DIR, "train.jsonl")
CACHE_PATH = os.path.join(OUT_DIR, "cache.json")

GRID_SIZE = 16
OBSTACLE_PROB = 0.18
NUM_SAMPLES = 2000
CELL_SIZE = 20

ACTIONS = {"UP", "DOWN", "LEFT", "RIGHT"}

MODEL_ID = "gpt-5.2"  # teacher

# =========================
# OpenAI client
# =========================
OpenAI_API_KEY = "your-api-key-here"
client = OpenAI(api_key=OpenAI_API_KEY)

# =========================
# Utilities
# =========================
def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)

def hash_scene(grid, start, goal) -> str:
    m = hashlib.sha256()
    m.update(grid.tobytes())
    m.update(bytes(start))
    m.update(bytes(goal))
    return m.hexdigest()

def load_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r") as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)

# =========================
# Scene generation
# =========================
def sample_scene() -> Tuple[np.ndarray, Tuple[int,int], Tuple[int,int]]:
    while True:
        grid = (np.random.rand(GRID_SIZE, GRID_SIZE) < OBSTACLE_PROB).astype(np.uint8)
        grid[0,:] = grid[-1,:] = grid[:,0] = grid[:,-1] = 0

        free = list(zip(*np.where(grid == 0)))
        if len(free) < 5:
            continue

        sy, sx = random.choice(free)
        gy, gx = random.choice(free)
        if (sx,sy) != (gx,gy):
            return grid, (sx,sy), (gx,gy)

def render_image(grid, start, goal, path):
    H, W = grid.shape
    img = Image.new("RGB", (W*CELL_SIZE, H*CELL_SIZE), "white")
    d = ImageDraw.Draw(img)

    for y in range(H):
        for x in range(W):
            if grid[y,x] == 1:
                d.rectangle(
                    [x*CELL_SIZE, y*CELL_SIZE,
                     (x+1)*CELL_SIZE-1, (y+1)*CELL_SIZE-1],
                    fill=(50,50,50)
                )

    sx, sy = start
    gx, gy = goal

    d.rectangle([sx*CELL_SIZE, sy*CELL_SIZE,
                 (sx+1)*CELL_SIZE-1, (sy+1)*CELL_SIZE-1],
                fill=(0,200,0))
    d.rectangle([gx*CELL_SIZE, gy*CELL_SIZE,
                 (gx+1)*CELL_SIZE-1, (gy+1)*CELL_SIZE-1],
                fill=(220,0,0))

    return img

# =========================
# Teacher prompt
# =========================
def build_prompt() -> str:
    return (
        "You are a robot with a top-down view of the environment.\n\n"
        "Tasks:\n"
        "1. Answer the spatial reasoning question.\n"
        "2. Produce a collision-free action plan.\n\n"
        "Rules:\n"
        "- Actions must be one of: UP DOWN LEFT RIGHT\n"
        "- No extra words in the action plan\n"
        "- Be concise and correct\n\n"
        "Return ONLY valid JSON in this format:\n"
        "{\n"
        '  "answer": "Yes or No",\n'
        '  "reasoning": "short explanation",\n'
        '  "actions": "UP RIGHT DOWN ..."\n'
        "}"
    )

# =========================
# Teacher call
# =========================
def query_teacher(image: Image.Image, question: str) -> Optional[Dict]:
    response = client.responses.create(
        model=MODEL_ID,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": build_prompt()},
                    {"type": "input_text", "text": f"Question: {question}"},
                    {"type": "input_image", "image": image},
                ],
            }
        ],
        temperature=0.0,
        max_output_tokens=300,
    )

    try:
        text = response.output_text
        data = json.loads(text)
        acts = data["actions"].split()
        if not all(a in ACTIONS for a in acts):
            return None
        return data
    except Exception:
        return None

# =========================
# Main loop
# =========================
def main():
    ensure_dirs()
    cache = load_cache()

    f_out = open(JSONL_PATH, "a", encoding="utf-8")

    for idx in tqdm(range(NUM_SAMPLES)):
        grid, start, goal = sample_scene()
        scene_id = hash_scene(grid, start, goal)

        if scene_id in cache:
            continue

        img = render_image(grid, start, goal, None)
        img_path = os.path.join(IMG_DIR, f"{scene_id}.png")
        img.save(img_path)

        question = "Can the robot reach the red goal without colliding with obstacles?"

        result = query_teacher(img, question)
        if result is None:
            continue

        record = {
            "image": img_path,
            "question": question,
            "answer": result["answer"],
            "teacher_reasoning": result["reasoning"],
            "teacher_actions": result["actions"],
        }

        f_out.write(json.dumps(record) + "\n")
        f_out.flush()

        cache[scene_id] = True
        save_cache(cache)

        time.sleep(0.5)  # polite rate limit

    f_out.close()

if __name__ == "__main__":
    main()
