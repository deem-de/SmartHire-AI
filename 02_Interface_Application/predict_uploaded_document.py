"""Predict entities from an uploaded PDF/DOCX/TXT file.

Pipeline:
document file -> raw text -> training-style JSON -> trained BERT model
-> extracted entities -> candidate profile JSON
"""

from __future__ import annotations

import argparse
from pathlib import Path

from candidate_profile import build_profile_from_entities, save_profile
from document_loader import extract_text_from_document
from export_profile_from_model import predict_entities
from json_formatter import text_to_resume_json
from models import get_device, load_tokenizer_for_model
from transformers import AutoModelForTokenClassification


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict resume entities from a document")
    parser.add_argument("--file_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="outputs/baseline_model")
    parser.add_argument("--output_path", type=str, default="outputs/uploaded_candidate_profile.json")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--section_aware", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Extracting raw text from document...")
    raw_text = extract_text_from_document(args.file_path)
    if not raw_text:
        raise ValueError("No text could be extracted from this document.")

    resume_json = text_to_resume_json(raw_text)

    print("Loading trained BERT model...")
    device = get_device()
    tokenizer = load_tokenizer_for_model(args.model_path)
    model = AutoModelForTokenClassification.from_pretrained(args.model_path).to(device)

    print("Predicting entities...")
    entities = predict_entities(
        resume_text=resume_json["content"],
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_length=args.max_length,
        section_aware=args.section_aware,
    )

    profile = build_profile_from_entities(entities)
    save_profile(profile, args.output_path)

    print(f"Uploaded document profile saved to: {Path(args.output_path).resolve()}")
    print("\nPredicted Candidate Profile")
    print("-" * 35)
    if not profile:
        print("No profile fields predicted. Try a clearer document or a better trained model.")
        return

    for field_name, values in profile.items():
        print(f"{field_name}: {', '.join(values[:5])}")


if __name__ == "__main__":
    main()
