"""Convert extracted document text into the training-style JSON format."""

from __future__ import annotations


def text_to_resume_json(raw_text: str) -> dict:
    """Create the same JSON structure used by the resume dataset.

    During prediction, annotation is empty because the model will predict labels.
    """

    return {
        "content": raw_text,
        "annotation": [],
        "extras": None,
    }
