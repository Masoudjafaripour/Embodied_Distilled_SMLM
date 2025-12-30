"""
Prompt templates for embodied spatial reasoning distillation.

This file defines:
- Teacher prompt (GPT-5.2)
- Student prompt (VLM / MLLM)
- Target formatting for supervision

All prompts are deterministic and minimal by design.
"""

# =========================
# Shared constants
# =========================
ACTION_VOCAB = "UP DOWN LEFT RIGHT"

# =========================
# Teacher (GPT-5.2) prompt
# =========================
def teacher_system_prompt() -> str:
    """
    System instruction for the teacher model.
    Enforces structure and output format.
    """
    return (
        "You are an expert robot planner with perfect spatial understanding.\n"
        "You see a top-down image of the environment.\n"
        "You must reason about space and produce correct actions.\n"
        "Follow all rules exactly."
    )

def teacher_user_prompt(question: str) -> str:
    """
    User prompt for the teacher model.
    The teacher must return JSON only.
    """
    return (
        "Tasks:\n"
        "1. Answer the spatial reasoning question.\n"
        "2. Produce a collision-free action plan.\n\n"
        "Rules:\n"
        f"- Actions must be one of: {ACTION_VOCAB}\n"
        "- Do NOT invent new actions\n"
        "- The plan must reach the goal if possible\n"
        "- Be concise and correct\n\n"
        "Return ONLY valid JSON in this exact format:\n"
        "{\n"
        '  "answer": "Yes or No",\n'
        '  "reasoning": "short explanation",\n'
        '  "actions": "UP RIGHT DOWN ..."\n'
        "}\n\n"
        f"Question: {question}"
    )

# =========================
# Student prompt
# =========================
def student_prompt(question: str) -> str:
    """
    Prompt used during student training and inference.
    Kept short to avoid language bias.
    """
    return (
        "You are a robot with a top-down view.\n"
        "Answer the spatial question and produce a collision-free action plan.\n"
        f"Use only actions: {ACTION_VOCAB}.\n\n"
        f"Question: {question}\n"
    )

# =========================
# Target formatting
# =========================
def format_target(answer: str, reasoning: str, actions: str) -> str:
    """
    Formats teacher outputs into a stable supervision string.
    This exact format is what the student learns to generate.
    """
    return (
        f"Answer: {answer}\n"
        f"Reasoning: {reasoning}\n"
        f"Actions: {actions}"
    )

# =========================
# Parsing helpers (optional)
# =========================
def is_valid_action_sequence(actions: str) -> bool:
    """
    Checks if an action sequence only contains allowed tokens.
    """
    tokens = actions.split()
    allowed = set(ACTION_VOCAB.split())
    return len(tokens) > 0 and all(t in allowed for t in tokens)
