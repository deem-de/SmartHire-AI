"""Run the full resume parsing experiment.

This script follows the project document:
1. Load annotated resume data
2. Convert character-level annotations to BIO labels
3. Train a baseline BERT NER model
4. Train a section-aware BERT NER model
5. Compare precision, recall, and F1
6. Print structured extracted entities
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, set_seed

from data_loading import load_resume_dataset, split_dataset
from evaluation import evaluate_model
from feature_engineering import print_section_summary
from models import MODEL_NAME, create_bert_ner_model, get_device, load_tokenizer
from preprocessing import (
    build_label_list,
    clean_text_for_display,
    decode_entities_from_labels,
    make_bert_features,
)
from training import create_data_loader, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resume Parsing with BERT NER")
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model_name",
        type=str,
        default=MODEL_NAME,
        help="Transformer checkpoint. Default keeps the project as BERT-based.",
    )
    parser.add_argument(
        "--class_weight_max",
        type=float,
        default=5.0,
        help="Maximum class weight. Lower values can reduce false positives.",
    )
    parser.add_argument(
        "--section_class_weight_max",
        type=float,
        default=None,
        help="Maximum class weight for section-aware model. Defaults to class_weight_max.",
    )
    parser.add_argument(
        "--no_class_weights",
        action="store_true",
        help="Disable weighted loss and use normal CrossEntropyLoss.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Use a small number of resumes for quick testing.",
    )
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Only build features and show dataset output. Useful before long training.",
    )
    return parser.parse_args()


def print_dataset_summary(records, label_list) -> None:
    """Print simple dataset information for the report and presentation."""

    total_entities = sum(len(record.entities) for record in records)
    lengths = [len(record.content) for record in records]

    print("\nDataset summary")
    print(f"- Resumes: {len(records)}")
    print(f"- Annotated entities: {total_entities}")
    print(f"- Entity labels: {len(label_list) // 2}")
    print(f"- Average resume length: {sum(lengths) / max(len(lengths), 1):.0f} characters")
    print(f"- Longest resume: {max(lengths) if lengths else 0} characters")


def print_metrics_table(baseline_metrics, section_metrics) -> None:
    """Print a clean comparison between baseline and section-aware model."""

    print("\nModel comparison")
    print("Metric                  Baseline      Section-aware")
    print("-" * 52)
    for metric in [
        "precision",
        "recall",
        "f1",
        "accuracy",
        "token_accuracy",
        "entity_token_accuracy",
        "entity_precision",
        "entity_recall",
        "entity_f1",
        "section_level_accuracy",
    ]:
        print(
            f"{metric:<22}"
            f"{baseline_metrics.get(metric, 0):>8.4f}"
            f"{section_metrics.get(metric, 0):>16.4f}"
        )


def show_gold_structured_output(record) -> None:
    """Print one resume as structured fields using the human annotations."""

    structured = {}
    for entity in record.entities:
        value = clean_text_for_display(entity.text)
        if value:
            structured.setdefault(entity.label, []).append(value)

    print("\nExample structured output from gold annotations")
    for label, values in structured.items():
        unique_values = list(dict.fromkeys(values))
        print(f"\n{label}:")
        for value in unique_values[:5]:
            print(f"- {value}")


def predict_one_resume(model_path, tokenizer, record, label_list, device, max_length) -> None:
    """Run a trained model on one resume and print predicted structured output."""

    id_to_label = {index: label for index, label in enumerate(label_list)}
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    encoded = tokenizer(
        record.content,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_offsets_mapping=False,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(
            input_ids=encoded["input_ids"].to(device),
            attention_mask=encoded["attention_mask"].to(device),
        )
        predictions = torch.argmax(outputs.logits, dim=-1)[0].cpu().tolist()

    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0].tolist())
    extracted = decode_entities_from_labels(tokens, predictions, id_to_label)

    print("\nExample structured output from model prediction")
    if not extracted:
        print(
            "No entities predicted. This usually means the model needs more training "
            "or is still predicting mostly O labels."
        )
        return

    for label, values in extracted.items():
        unique_values = list(dict.fromkeys(values))
        print(f"\n{label}:")
        for value in unique_values[:5]:
            print(f"- {value}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    device = get_device()
    section_class_weight_max = (
        args.section_class_weight_max
        if args.section_class_weight_max is not None
        else args.class_weight_max
    )

    print("Loading resume dataset...")
    records = load_resume_dataset(args.dataset_path)
    if args.sample_size:
        sample_count = min(args.sample_size, len(records))
        records = random.Random(args.seed).sample(records, sample_count)

    label_list = build_label_list(records)
    label_to_id = {label: index for index, label in enumerate(label_list)}
    id_to_label = {index: label for index, label in enumerate(label_list)}

    print_dataset_summary(records, label_list)
    print_section_summary(records)

    train_records, validation_records, test_records = split_dataset(records, seed=args.seed)

    print("\nLoading BERT tokenizer...")
    tokenizer = load_tokenizer(args.model_name)

    print("Building baseline features...")
    baseline_train = make_bert_features(
        train_records, tokenizer, label_to_id, args.max_length, section_aware=False
    )
    baseline_validation = make_bert_features(
        validation_records, tokenizer, label_to_id, args.max_length, section_aware=False
    )
    baseline_test = make_bert_features(
        test_records, tokenizer, label_to_id, args.max_length, section_aware=False
    )

    print("Building section-aware features...")
    section_train = make_bert_features(
        train_records, tokenizer, label_to_id, args.max_length, section_aware=True
    )
    section_validation = make_bert_features(
        validation_records, tokenizer, label_to_id, args.max_length, section_aware=True
    )
    section_test = make_bert_features(
        test_records, tokenizer, label_to_id, args.max_length, section_aware=True
    )

    print(f"\nBaseline training chunks: {len(baseline_train)}")
    print(f"Section-aware training chunks: {len(section_train)}")

    if args.skip_training:
        print("\nTraining skipped. This confirms loading and preprocessing work.")
        show_gold_structured_output(records[0])
        return

    print("\nTraining baseline model...")
    baseline_model = create_bert_ner_model(label_list, args.model_name)
    train_model(
        baseline_model,
        baseline_train,
        baseline_validation,
        device,
        id_to_label,
        output_dir / "baseline_model",
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_class_weights=not args.no_class_weights,
        class_weight_max=args.class_weight_max,
    )
    tokenizer.save_pretrained(output_dir / "baseline_model")

    print("\nTraining section-aware model...")
    section_model = create_bert_ner_model(label_list, args.model_name)
    train_model(
        section_model,
        section_train,
        section_validation,
        device,
        id_to_label,
        output_dir / "section_aware_model",
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_class_weights=not args.no_class_weights,
        class_weight_max=section_class_weight_max,
    )
    tokenizer.save_pretrained(output_dir / "section_aware_model")

    print("\nEvaluating on test set...")
    baseline_model = AutoModelForTokenClassification.from_pretrained(
        str(output_dir / "baseline_model")
    ).to(device)
    section_model = AutoModelForTokenClassification.from_pretrained(
        str(output_dir / "section_aware_model")
    ).to(device)

    baseline_loader = create_data_loader(baseline_test, batch_size=args.batch_size)
    section_loader = create_data_loader(section_test, batch_size=args.batch_size)

    baseline_metrics = evaluate_model(
        baseline_model,
        baseline_loader,
        device,
        id_to_label,
        show_distribution=True,
    )
    section_metrics = evaluate_model(
        section_model,
        section_loader,
        device,
        id_to_label,
        show_distribution=True,
    )

    print_metrics_table(baseline_metrics, section_metrics)
    metrics_path = output_dir / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(
            {
                "baseline": baseline_metrics,
                "section_aware": section_metrics,
                "seed": args.seed,
                "model_name": args.model_name,
                "epochs": args.epochs,
                "max_length": args.max_length,
                "batch_size": args.batch_size,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nMetrics saved to: {metrics_path.resolve()}")
    show_gold_structured_output(test_records[0])

    best_model_name = (
        "section_aware_model"
        if section_metrics.get("f1", 0) >= baseline_metrics.get("f1", 0)
        else "baseline_model"
    )
    print(f"\nUsing best F1 model for sample prediction: {best_model_name}")
    predict_one_resume(
        output_dir / best_model_name,
        tokenizer,
        test_records[0],
        label_list,
        device,
        args.max_length,
    )


if __name__ == "__main__":
    main()
