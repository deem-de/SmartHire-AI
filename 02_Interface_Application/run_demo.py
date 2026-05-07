"""One-command demo runner for the resume parsing project.

This script connects:
uploaded document -> BERT profile extraction -> AI interview assistant
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from candidate_profile import save_profile
from document_loader import extract_text_from_document
from export_profile_from_model import predict_entities
from json_formatter import text_to_resume_json
from models import get_device, load_tokenizer_for_model
from transformers import AutoModelForTokenClassification


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full resume demo")
    parser.add_argument("--file_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="outputs/baseline_model")
    parser.add_argument("--profile_path", type=str, default="outputs/demo_candidate_profile.json")
    parser.add_argument("--language", type=str, default="English", choices=["English", "Arabic"])
    parser.add_argument("--questions", type=int, default=3)
    parser.add_argument("--offline", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Step 1: Extracting text from uploaded document...")
    raw_text = extract_text_from_document(args.file_path)
    resume_json = text_to_resume_json(raw_text)

    print("Step 2: Loading trained BERT model...")
    device = get_device()
    tokenizer = load_tokenizer_for_model(args.model_path)
    model = AutoModelForTokenClassification.from_pretrained(args.model_path).to(device)

    print("Step 3: Predicting resume entities...")
    entities = predict_entities(
        resume_text=resume_json["content"],
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_length=256,
        section_aware=False,
    )

    from candidate_profile import build_profile_from_entities

    profile = build_profile_from_entities(entities)
    save_profile(profile, args.profile_path)

    print(f"Candidate profile saved to: {Path(args.profile_path).resolve()}")

    print("Step 4: Starting interview assistant...")
    command = [
        sys.executable,
        "interview_terminal.py",
        "--profile_path",
        args.profile_path,
        "--language",
        args.language,
        "--questions",
        str(args.questions),
    ]
    if args.offline:
        command.append("--offline")

    subprocess.run(command, check=False)


if __name__ == "__main__":
    main()
