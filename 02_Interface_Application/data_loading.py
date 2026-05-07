"""Load the resume NER dataset and create train/validation/test splits.

The project document says the dataset is a JSON file where every resume has:
- content: the full resume text
- annotation: character-level entity labels
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from sklearn.model_selection import train_test_split


# UNKNOWN/OTHER are too general for resume parsing output. Skipping them keeps
# the model focused on useful fields such as skills, education, companies, etc.
SKIP_LABELS = {"UNKNOWN", "OTHER"}


@dataclass
class Entity:
    """One annotated entity from the resume."""

    start: int
    end: int
    label: str
    text: str


@dataclass
class ResumeRecord:
    """One resume with its text and entity annotations."""

    content: str
    entities: list[Entity]


def default_dataset_path() -> Path:
    """Find the dataset in common student project locations."""

    possible_paths = [
        Path("data") / "resume_ner_training_data.json",
        Path("data") / "Entity Recognition in Resumes.json",
        Path("data") / "Entity Recognition in Resumes (1).json",
        Path.home() / "Downloads" / "Entity Recognition in Resumes.json",
        Path.home() / "Downloads" / "Entity Recognition in Resumes (1).json",
    ]

    for path in possible_paths:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Dataset not found. Put the JSON file in the data folder or pass --dataset_path."
    )


def _read_json_or_jsonl(path: Path) -> list[dict]:
    """Read either normal JSON or JSON-lines format."""

    text = path.read_text(encoding="utf-8")

    try:
        data = json.loads(text)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        records = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
        return records


def _parse_entities(annotation: list[dict], content: str = "") -> list[Entity]:
    """Convert raw annotation objects into simple Entity objects.

    This is safe preprocessing for NER: we do not change the resume text, because
    changing text would break character offsets. Instead, we clean noisy labels
    and trim whitespace around entity spans.
    """

    entities: list[Entity] = []

    for item in annotation or []:
        labels = item.get("label", [])
        label = labels[0] if labels else "UNKNOWN"
        if label in SKIP_LABELS:
            continue

        for point in item.get("points", []):
            start = int(point["start"])
            end = int(point["end"])
            start, end = _trim_entity_span(content, start, end)
            if start > end:
                continue

            text = content[start : end + 1] if content else point.get("text", "").strip()
            if not text:
                continue

            entities.append(Entity(start=start, end=end, label=label, text=text))

    return _remove_overlapping_entities(entities)


def _trim_entity_span(content: str, start: int, end: int) -> tuple[int, int]:
    """Trim whitespace around an entity span without changing the resume text."""

    if not content:
        return start, end

    start = max(start, 0)
    end = min(end, len(content) - 1)

    while start <= end and content[start].isspace():
        start += 1
    while end >= start and content[end].isspace():
        end -= 1

    return start, end


def _remove_overlapping_entities(entities: list[Entity]) -> list[Entity]:
    """Remove overlapping spans because BIO tagging supports one label per token.

    If two annotations overlap, keeping both creates noisy training labels. We keep
    the shorter span because it is usually a more precise entity.
    """

    selected_entities: list[Entity] = []
    precise_first = sorted(
        entities,
        key=lambda entity: (entity.end - entity.start, entity.start),
    )

    for entity in precise_first:
        overlaps_existing = any(
            entity.start <= selected.end and entity.end >= selected.start
            for selected in selected_entities
        )

        if not overlaps_existing:
            selected_entities.append(entity)

    return sorted(selected_entities, key=lambda entity: entity.start)


def load_resume_dataset(dataset_path: str | Path | None = None) -> list[ResumeRecord]:
    """Load resumes from the dataset file."""

    path = Path(dataset_path) if dataset_path else default_dataset_path()
    raw_records = _read_json_or_jsonl(path)

    records: list[ResumeRecord] = []
    for item in raw_records:
        content = item.get("content", "")
        entities = _parse_entities(item.get("annotation", []), content)
        records.append(ResumeRecord(content=content, entities=entities))

    return records


def split_dataset(
    records: list[ResumeRecord],
    seed: int = 42,
) -> tuple[list[ResumeRecord], list[ResumeRecord], list[ResumeRecord]]:
    """Split the dataset into 70% training, 15% validation, and 15% test."""

    train_records, temp_records = train_test_split(
        records, test_size=0.30, random_state=seed, shuffle=True
    )
    validation_records, test_records = train_test_split(
        temp_records, test_size=0.50, random_state=seed, shuffle=True
    )
    return train_records, validation_records, test_records
