"""Simple evaluation metrics for resume NER."""

from __future__ import annotations

from collections import Counter

import torch
from sklearn.metrics import precision_recall_fscore_support


def compute_token_metrics(
    true_label_ids: list[int],
    predicted_label_ids: list[int],
    id_to_label: dict[int, str],
) -> dict[str, float]:
    """Compute token-level precision, recall, and F1.

    Tokens with label -100 are ignored because they are padding or special tokens.
    """

    filtered_true = []
    filtered_predicted = []

    for true_id, predicted_id in zip(true_label_ids, predicted_label_ids):
        if true_id == -100:
            continue
        filtered_true.append(id_to_label[int(true_id)])
        filtered_predicted.append(id_to_label[int(predicted_id)])

    labels_without_o = [label for label in id_to_label.values() if label != "O"]

    precision, recall, f1, _ = precision_recall_fscore_support(
        filtered_true,
        filtered_predicted,
        labels=labels_without_o,
        average="micro",
        zero_division=0,
    )

    correct_tokens = sum(
        true_label == predicted_label
        for true_label, predicted_label in zip(filtered_true, filtered_predicted)
    )
    token_accuracy = correct_tokens / len(filtered_true) if filtered_true else 0.0

    entity_pairs = [
        (true_label, predicted_label)
        for true_label, predicted_label in zip(filtered_true, filtered_predicted)
        if true_label != "O"
    ]
    entity_correct = sum(true_label == predicted_label for true_label, predicted_label in entity_pairs)
    entity_token_accuracy = entity_correct / len(entity_pairs) if entity_pairs else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(token_accuracy),
        "token_accuracy": float(token_accuracy),
        "entity_token_accuracy": float(entity_token_accuracy),
    }


def bio_to_entity_spans(label_ids: list[int], id_to_label: dict[int, str]) -> set[tuple[int, int, str]]:
    """Convert BIO label IDs into entity spans for exact-match evaluation."""

    spans = set()
    start = None
    current_label = None

    for index, label_id in enumerate(label_ids + [-100]):
        label = id_to_label.get(int(label_id), "O") if label_id != -100 else "O"

        if label == "O":
            if current_label is not None:
                spans.add((start, index - 1, current_label))
            start = None
            current_label = None
            continue

        prefix, entity_label = label.split("-", 1)
        if prefix == "B" or entity_label != current_label:
            if current_label is not None:
                spans.add((start, index - 1, current_label))
            start = index
            current_label = entity_label

    return spans


def compute_entity_exact_match(
    true_sequences: list[list[int]],
    predicted_sequences: list[list[int]],
    id_to_label: dict[int, str],
) -> dict[str, float]:
    """Compute entity-level exact-match precision, recall, and F1."""

    true_entities = set()
    predicted_entities = set()

    for sequence_index, (true_ids, predicted_ids) in enumerate(
        zip(true_sequences, predicted_sequences)
    ):
        valid_pairs = [
            (true_id, predicted_id)
            for true_id, predicted_id in zip(true_ids, predicted_ids)
            if true_id != -100
        ]
        true_ids = [true_id for true_id, _ in valid_pairs]
        predicted_ids = [predicted_id for _, predicted_id in valid_pairs]

        for span in bio_to_entity_spans(true_ids, id_to_label):
            true_entities.add((sequence_index, *span))
        for span in bio_to_entity_spans(predicted_ids, id_to_label):
            predicted_entities.add((sequence_index, *span))

    correct = len(true_entities & predicted_entities)
    precision = correct / len(predicted_entities) if predicted_entities else 0.0
    recall = correct / len(true_entities) if true_entities else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return {
        "entity_precision": precision,
        "entity_recall": recall,
        "entity_f1": f1,
    }


def print_prediction_distribution(
    true_label_ids: list[int],
    predicted_label_ids: list[int],
    id_to_label: dict[int, str],
) -> None:
    """Print a small report showing what the model predicts.

    This is useful for debugging class imbalance. If most predictions are O,
    the model is not learning the entity labels yet.
    """

    true_counts = Counter(id_to_label[int(label_id)] for label_id in true_label_ids)
    predicted_counts = Counter(
        id_to_label[int(label_id)] for label_id in predicted_label_ids
    )

    print("\nTrue vs predicted label distribution")
    print("Label                          True   Predicted")
    print("-" * 54)
    for label_id in sorted(id_to_label):
        label = id_to_label[label_id]
        print(
            f"{label:<28}"
            f"{true_counts.get(label, 0):>8}"
            f"{predicted_counts.get(label, 0):>12}"
        )


def evaluate_model(
    model,
    data_loader,
    device,
    id_to_label: dict[int, str],
    show_distribution: bool = False,
) -> dict[str, float]:
    """Run the model on a dataset and return clear evaluation metrics."""

    model.eval()
    all_true = []
    all_predicted = []
    true_sequences = []
    predicted_sequences = []
    section_correct = {}
    section_total = {}

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)

            section_names = batch.get("section_name", ["full_resume"] * len(labels))
            for true_sequence, predicted_sequence, section_name in zip(
                labels.cpu(), predictions.cpu(), section_names
            ):
                true_list = true_sequence.tolist()
                predicted_list = predicted_sequence.tolist()
                true_sequences.append(true_list)
                predicted_sequences.append(predicted_list)

                for true_id, predicted_id in zip(true_list, predicted_list):
                    if true_id != -100:
                        all_true.append(true_id)
                        all_predicted.append(predicted_id)
                        section_total[section_name] = section_total.get(section_name, 0) + 1
                        if true_id == predicted_id:
                            section_correct[section_name] = (
                                section_correct.get(section_name, 0) + 1
                            )

    token_metrics = compute_token_metrics(all_true, all_predicted, id_to_label)
    if show_distribution:
        print_prediction_distribution(all_true, all_predicted, id_to_label)

    entity_metrics = compute_entity_exact_match(
        true_sequences, predicted_sequences, id_to_label
    )
    section_accuracies = [
        section_correct.get(section_name, 0) / total
        for section_name, total in section_total.items()
        if total > 0
    ]
    section_level_accuracy = (
        sum(section_accuracies) / len(section_accuracies) if section_accuracies else 0.0
    )

    return {
        **token_metrics,
        **entity_metrics,
        "section_level_accuracy": section_level_accuracy,
    }
