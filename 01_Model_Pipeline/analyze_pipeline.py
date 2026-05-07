"""Quick dataset audit before training.

This script helps students check whether the resume dataset is loaded correctly
before starting a long BERT training run.
"""

from __future__ import annotations

import argparse
from collections import Counter

from data_loading import load_resume_dataset
from feature_engineering import print_section_summary
from preprocessing import build_label_list


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze resume NER dataset")
    parser.add_argument("--dataset_path", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_resume_dataset(args.dataset_path)
    label_list = build_label_list(records)

    label_counts = Counter()
    lengths = []
    for record in records:
        lengths.append(len(record.content))
        for entity in record.entities:
            label_counts[entity.label] += 1

    print("Dataset audit")
    print("-" * 40)
    print(f"Resumes: {len(records)}")
    print(f"Entity labels: {len(label_list) // 2}")
    print(f"Annotated entities: {sum(label_counts.values())}")
    print(f"Average resume length: {sum(lengths) / max(len(lengths), 1):.0f} characters")
    print(f"Longest resume: {max(lengths) if lengths else 0} characters")

    print("\nLabel distribution")
    print("-" * 40)
    for label, count in label_counts.most_common():
        print(f"{label:<18} {count}")

    print_section_summary(records)


if __name__ == "__main__":
    main()
