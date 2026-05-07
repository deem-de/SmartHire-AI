"""Export a candidate profile using the trained BERT NER model.

This connects Part 1 to Part 2:
trained BERT model -> predicted entities -> candidate profile JSON -> interview assistant
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification

from candidate_profile import build_profile_from_entities, save_profile
from data_loading import load_resume_dataset
from models import get_device, load_tokenizer_for_model
from preprocessing import decode_entities_from_labels, split_resume_into_sections, TextSegment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export profile from trained BERT model")
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--resume_index", type=int, default=0)
    parser.add_argument("--text_path", type=str, default=None)
    parser.add_argument("--model_path", type=str, default="outputs/baseline_model")
    parser.add_argument("--output_path", type=str, default="outputs/model_candidate_profile.json")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument(
        "--section_aware",
        action="store_true",
        help="Split the resume into sections before prediction.",
    )
    return parser.parse_args()


def load_resume_text(dataset_path: str | None, resume_index: int, text_path: str | None) -> str:
    """Load resume text from either a text file or the JSON dataset."""

    if text_path:
        return Path(text_path).read_text(encoding="utf-8")

    records = load_resume_dataset(dataset_path)
    if resume_index < 0 or resume_index >= len(records):
        raise IndexError(f"resume_index must be between 0 and {len(records) - 1}.")
    return records[resume_index].content


def get_id_to_label(model) -> dict[int, str]:
    """Read BIO labels from the saved model config."""

    id_to_label = {}
    for key, value in model.config.id2label.items():
        id_to_label[int(key)] = value
    return id_to_label


def predict_entities(
    resume_text: str,
    model,
    tokenizer,
    device,
    max_length: int = 256,
    section_aware: bool = False,
) -> dict[str, list[str]]:
    """Predict entities from resume text using the trained model."""

    id_to_label = get_id_to_label(model)
    extracted: dict[str, list[str]] = {}

    segments = (
        split_resume_into_sections(resume_text)
        if section_aware
        else [TextSegment(text=resume_text, start_offset=0, section_name="full_resume")]
    )

    model.eval()

    for segment in segments:
        encoded_chunks = tokenizer(
            segment.text,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_overflowing_tokens=True,
            stride=64,
            return_tensors="pt",
        )

        chunk_count = encoded_chunks["input_ids"].shape[0]
        for chunk_index in range(chunk_count):
            input_ids = encoded_chunks["input_ids"][chunk_index : chunk_index + 1].to(device)
            attention_mask = encoded_chunks["attention_mask"][chunk_index : chunk_index + 1].to(device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=-1)[0].cpu().tolist()

            tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())
            chunk_entities = decode_entities_from_labels(tokens, predictions, id_to_label)

            for label, values in chunk_entities.items():
                for value in values:
                    clean_value = " ".join(value.split())
                    if clean_value and clean_value not in extracted.setdefault(label, []):
                        extracted[label].append(clean_value)

    return extracted


def main() -> None:
    args = parse_args()
    device = get_device()

    resume_text = load_resume_text(args.dataset_path, args.resume_index, args.text_path)
    tokenizer = load_tokenizer_for_model(args.model_path)
    model = AutoModelForTokenClassification.from_pretrained(args.model_path).to(device)

    entities = predict_entities(
        resume_text=resume_text,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_length=args.max_length,
        section_aware=args.section_aware,
    )

    profile = build_profile_from_entities(entities)
    save_profile(profile, args.output_path)

    print(f"Model profile saved to: {Path(args.output_path).resolve()}")
    print("\nPredicted Candidate Profile")
    print("-" * 35)
    if not profile:
        print("No profile fields predicted. Try the baseline model or train for more epochs.")
        return

    for field_name, values in profile.items():
        print(f"{field_name}: {', '.join(values[:5])}")


if __name__ == "__main__":
    main()
