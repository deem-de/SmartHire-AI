"""BERT models used in the resume parsing project."""

from __future__ import annotations

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer


MODEL_NAME = "bert-base-cased"


def load_tokenizer(model_name: str = MODEL_NAME):
    """Load the BERT tokenizer used for token-level NER."""

    return AutoTokenizer.from_pretrained(model_name)


def load_tokenizer_for_model(model_path: str, fallback_model_name: str = MODEL_NAME):
    """Load tokenizer from the trained model folder, with a BERT fallback.

    Saving and reloading the tokenizer keeps the demo connected to the exact
    model checkpoint used during training.
    """

    try:
        return AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    except Exception:
        return AutoTokenizer.from_pretrained(fallback_model_name)


def create_bert_ner_model(
    label_list: list[str],
    model_name: str = MODEL_NAME,
):
    """Create a BERT token-classification model.

    The same function is used for:
    - baseline model
    - section-aware model
    """

    id_to_label = {index: label for index, label in enumerate(label_list)}
    label_to_id = {label: index for index, label in id_to_label.items()}

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label_list),
        id2label=id_to_label,
        label2id=label_to_id,
    )
    return model


def get_device() -> torch.device:
    """Use GPU if available, otherwise CPU."""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
