"""Simple BERT fine-tuning loop for resume NER."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from evaluation import evaluate_model


class ResumeNERDataset(Dataset):
    """A small PyTorch dataset for BERT token classification."""

    def __init__(self, features: list[dict]):
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict:
        item = self.features[index]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(item["labels"], dtype=torch.long),
            "section_name": item.get("section_name", "full_resume"),
        }


def create_data_loader(
    features: list[dict],
    batch_size: int = 16,
    shuffle: bool = False,
) -> DataLoader:
    """Create a PyTorch DataLoader."""

    dataset = ResumeNERDataset(features)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def compute_class_weights(
    features: list[dict],
    id_to_label: dict[int, str],
    max_weight: float = 5.0,
) -> torch.Tensor:
    """Compute simple class weights from the training BIO labels.

    Class imbalance matters in NER because common labels can dominate the loss.
    Higher weights give rare labels more importance during training.
    Padding labels (-100) are ignored because they are not real classes.
    """

    num_labels = len(id_to_label)
    label_counts = torch.zeros(num_labels, dtype=torch.float)

    for feature in features:
        for label_id in feature["labels"]:
            if label_id != -100:
                label_counts[int(label_id)] += 1

    total_labels = label_counts.sum().item()
    class_weights = torch.ones(num_labels, dtype=torch.float)

    if total_labels == 0:
        print("\nNo training labels found. Using equal class weights.")
        return class_weights

    for label_id in range(num_labels):
        count = label_counts[label_id].item()
        if count > 0:
            class_weights[label_id] = total_labels / (num_labels * count)
        else:
            # If a label does not appear in this split, keep a safe default weight.
            class_weights[label_id] = 1.0

    # Very rare labels can produce huge weights, so this cap keeps training stable.
    # A small cap is easier to explain and safer for a student project.
    class_weights = torch.clamp(class_weights, min=0.2, max=max_weight)

    print("\nTraining label distribution and class weights")
    print("Label                          Count      Weight")
    print("-" * 52)
    for label_id in range(num_labels):
        print(
            f"{id_to_label[label_id]:<28}"
            f"{int(label_counts[label_id].item()):>7}"
            f"{class_weights[label_id].item():>12.4f}"
        )

    print(
        "\nNote: class weights help rare labels, but imbalance may still affect performance."
    )
    return class_weights


def train_model(
    model,
    train_features: list[dict],
    validation_features: list[dict],
    device,
    id_to_label: dict[int, str],
    output_dir: str | Path,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    patience: int = 2,
    use_class_weights: bool = True,
    class_weight_max: float = 5.0,
) -> dict[str, float]:
    """Fine-tune BERT and keep the best model based on validation F1."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_loader = create_data_loader(train_features, batch_size=batch_size, shuffle=True)
    validation_loader = create_data_loader(
        validation_features, batch_size=batch_size, shuffle=False
    )

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    if use_class_weights:
        class_weights = compute_class_weights(
            train_features,
            id_to_label,
            max_weight=class_weight_max,
        ).to(device)
        loss_function = torch.nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=-100,
        )
    else:
        print("\nClass weights disabled. Using normal CrossEntropyLoss.")
        loss_function = torch.nn.CrossEntropyLoss(ignore_index=-100)

    best_f1 = -1.0
    best_metrics: dict[str, float] = {}
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits
            loss = loss_function(
                logits.view(-1, logits.shape[-1]),
                labels.view(-1),
            )
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())

        average_loss = total_loss / max(len(train_loader), 1)
        metrics = evaluate_model(model, validation_loader, device, id_to_label)

        print(
            f"Epoch {epoch}: loss={average_loss:.4f}, "
            f"val_precision={metrics['precision']:.4f}, "
            f"val_recall={metrics['recall']:.4f}, "
            f"val_f1={metrics['f1']:.4f}"
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_metrics = metrics
            epochs_without_improvement = 0
            model.save_pretrained(output_path)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping: validation F1 did not improve.")
            break

    return best_metrics
