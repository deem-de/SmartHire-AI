"""Prepare the new Resume NER dataset in the same JSON format as the old dataset.

This keeps the project requirements unchanged:
- content: raw resume text
- annotation: character-level entity spans
- extras: null

The source dataset is token-level BIO, so this script reconstructs text and
converts BIO labels into character spans.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

from datasets import load_dataset


DEFAULT_DATASET_NAME = "yashpwr/resume-ner-training-data"
DEFAULT_OUTPUT_PATH = Path("data") / "resume_ner_training_data.json"
SURROGATE_PATTERN = re.compile(r"[\ud800-\udfff]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert new Resume NER dataset to project JSON format")
    parser.add_argument("--dataset_name", type=str, default=DEFAULT_DATASET_NAME)
    parser.add_argument(
        "--local_path",
        type=str,
        default=None,
        help="Optional local JSON/JSONL/CSV/Parquet file instead of a Hugging Face dataset.",
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output_path", type=str, default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--tokens_column", type=str, default="tokens")
    parser.add_argument("--tags_column", type=str, default="ner_tags")
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--annotations_column", type=str, default="annotations")
    parser.add_argument(
        "--inspect_only",
        action="store_true",
        help="Only print dataset columns and a small example, then stop.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Optional smaller sample for quick testing.",
    )
    return parser.parse_args()


def load_source_dataset(dataset_name: str, split: str, local_path: str | None):
    """Load either the online dataset or a local file."""

    if local_path:
        path = Path(local_path)
        suffix = path.suffix.lower()

        if suffix in {".json", ".jsonl"}:
            if suffix == ".jsonl":
                records = []
                with path.open("r", encoding="utf-8") as file:
                    for line in file:
                        line = line.strip()
                        if line:
                            records.append(json.loads(line))
                return records

            with path.open("r", encoding="utf-8") as file:
                data = json.load(file)
            if isinstance(data, dict):
                for value in data.values():
                    if isinstance(value, list):
                        return value
                return [data]
            return data
        if suffix == ".csv":
            return load_dataset("csv", data_files=str(path), split="train")
        if suffix == ".parquet":
            return load_dataset("parquet", data_files=str(path), split="train")

        raise ValueError("Local dataset must be .json, .jsonl, .csv, or .parquet")

    return load_dataset(dataset_name, split=split)


def get_columns(dataset_split) -> list[str]:
    """Return column names from either a Hugging Face Dataset or a Python list."""

    if hasattr(dataset_split, "column_names"):
        return list(dataset_split.column_names)
    if dataset_split:
        return list(dataset_split[0].keys())
    return []


def get_label_names(dataset_split, tags_column: str) -> list[str] | None:
    """Read label names from the Hugging Face dataset schema if available."""

    if not hasattr(dataset_split, "features"):
        return None

    feature = dataset_split.features.get(tags_column)
    if feature is None:
        return None

    tag_feature = getattr(feature, "feature", None)
    names = getattr(tag_feature, "names", None)
    return list(names) if names else None


def parse_list_value(value):
    """Allow local CSV files to store tokens/tags as string lists."""

    if isinstance(value, list):
        return value

    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            return ast.literal_eval(stripped)
        return stripped.split()

    return value


def clean_text_value(value: str) -> str:
    """Remove broken surrogate characters that cannot be saved as UTF-8."""

    return SURROGATE_PATTERN.sub("", str(value))


def tag_to_label(tag, label_names: list[str] | None) -> str:
    """Convert a numeric or string BIO tag into a label string."""

    if isinstance(tag, int) and label_names:
        return label_names[tag]
    return str(tag)


def rebuild_text_and_offsets(tokens: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """Join tokens into text and record each token's character span."""

    content_parts = []
    offsets = []
    cursor = 0

    for token in tokens:
        token = clean_text_value(token)

        if content_parts:
            content_parts.append(" ")
            cursor += 1

        start = cursor
        content_parts.append(token)
        cursor += len(token)
        end = cursor - 1
        offsets.append((start, end))

    return "".join(content_parts), offsets


def bio_to_annotations(tokens: list[str], tags: list, label_names: list[str] | None) -> tuple[str, list[dict]]:
    """Convert BIO token labels into character-level annotation objects."""

    content, offsets = rebuild_text_and_offsets(tokens)
    annotations = []
    current_label = None
    current_start = None
    current_end = None

    def close_entity() -> None:
        nonlocal current_label, current_start, current_end

        if current_label is None:
            return

        entity_text = content[current_start : current_end + 1]
        annotations.append(
            {
                "label": [current_label],
                "points": [
                    {
                        "start": current_start,
                        "end": current_end,
                        "text": entity_text,
                    }
                ],
            }
        )
        current_label = None
        current_start = None
        current_end = None

    for index, tag in enumerate(tags):
        label = tag_to_label(tag, label_names)
        token_start, token_end = offsets[index]

        if label == "O":
            close_entity()
            continue

        if "-" in label:
            prefix, entity_label = label.split("-", 1)
        else:
            prefix, entity_label = "B", label

        if prefix == "B" or current_label != entity_label:
            close_entity()
            current_label = entity_label
            current_start = token_start
            current_end = token_end
        else:
            current_end = token_end

    close_entity()
    return content, annotations


def print_dataset_preview(dataset_split) -> None:
    """Show columns so we can verify the dataset format before training."""

    print("Dataset columns:")
    for column in get_columns(dataset_split):
        print(f"- {column}")

    if len(dataset_split) == 0:
        print("Dataset is empty.")
        return

    first_row = dataset_split[0]
    print("\nFirst row preview:")
    for key, value in first_row.items():
        text = str(value)
        if len(text) > 250:
            text = text[:250] + "..."
        print(f"{key}: {text}")


def validate_ner_columns(dataset_split, tokens_column: str, tags_column: str) -> None:
    """Make sure the dataset really has token-level NER labels."""

    columns = set(get_columns(dataset_split))
    missing = [name for name in [tokens_column, tags_column] if name not in columns]

    if not missing:
        return

    print_dataset_preview(dataset_split)
    raise ValueError(
        "\nThis dataset does not contain the token-level NER columns needed for BERT training.\n"
        f"Missing columns: {missing}\n"
        f"Found columns: {get_columns(dataset_split)}\n\n"
        "For this project, we need supervised NER data with columns like:\n"
        "- tokens: ['John', 'knows', 'Python']\n"
        "- ner_tags: ['B-Name', 'O', 'B-Skills'] or numeric tag IDs\n\n"
        "The downloaded Hugging Face version appears to be a different format, such as messages/chat data.\n"
        "Use a local Kaggle file that contains tokens and ner_tags, then run:\n"
        "python prepare_new_dataset.py --local_path \"C:\\path\\to\\dataset.json\" --output_path data/resume_ner_training_data.json\n"
    )


def has_span_annotations(dataset_split, text_column: str, annotations_column: str) -> bool:
    """Check for the new Kaggle format: text plus character-span annotations."""

    columns = set(get_columns(dataset_split))
    return text_column in columns and annotations_column in columns


def span_annotations_to_project_format(row: dict, text_column: str, annotations_column: str) -> dict:
    """Convert [start, end, label] annotations into the project JSON format.

    The new Kaggle data uses end-exclusive spans, so we store end - 1 because
    the original project loader expects inclusive character spans.
    """

    content = clean_text_value(row.get(text_column, ""))
    converted_annotations = []

    for annotation in row.get(annotations_column, []) or []:
        if isinstance(annotation, dict):
            start = int(annotation.get("start", 0))
            end = int(annotation.get("end", 0))
            label = annotation.get("label", "UNKNOWN")
        else:
            start = int(annotation[0])
            end = int(annotation[1])
            label = annotation[2]

        inclusive_end = max(start, end - 1)
        entity_text = clean_text_value(content[start : inclusive_end + 1])
        if not entity_text.strip():
            continue

        converted_annotations.append(
            {
                "label": [str(label)],
                "points": [
                    {
                        "start": start,
                        "end": inclusive_end,
                        "text": entity_text,
                    }
                ],
            }
        )

    return {
        "content": content,
        "annotation": converted_annotations,
        "extras": None,
    }


def convert_dataset(
    dataset_name: str,
    split: str,
    output_path: str,
    sample_size: int | None,
    local_path: str | None,
    tokens_column: str,
    tags_column: str,
    text_column: str,
    annotations_column: str,
    inspect_only: bool,
) -> None:
    """Download and convert the dataset."""

    source_name = local_path if local_path else f"{dataset_name} [{split}]"
    print(f"Loading dataset: {source_name}")
    dataset_split = load_source_dataset(dataset_name, split, local_path)

    if inspect_only:
        print_dataset_preview(dataset_split)
        return

    if sample_size:
        sample_count = min(sample_size, len(dataset_split))
        if hasattr(dataset_split, "select"):
            dataset_split = dataset_split.select(range(sample_count))
        else:
            dataset_split = dataset_split[:sample_count]

    records = []

    if has_span_annotations(dataset_split, text_column, annotations_column):
        print("Detected format: text + character-span annotations")
        for row in dataset_split:
            records.append(span_annotations_to_project_format(row, text_column, annotations_column))
    else:
        print("Detected format: token-level BIO labels")
        validate_ner_columns(dataset_split, tokens_column, tags_column)
        label_names = get_label_names(dataset_split, tags_column)

        for row in dataset_split:
            tokens = parse_list_value(row[tokens_column])
            tags = parse_list_value(row[tags_column])
            if len(tokens) != len(tags):
                raise ValueError(
                    "Token/tag length mismatch. "
                    f"tokens={len(tokens)}, tags={len(tags)}. "
                    "Please check the dataset format."
                )
            content, annotations = bio_to_annotations(tokens, tags, label_names)
            records.append(
                {
                    "content": content,
                    "annotation": annotations,
                    "extras": None,
                }
            )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Converted records: {len(records)}")
    print(f"Saved to: {output.resolve()}")


def main() -> None:
    args = parse_args()
    convert_dataset(
        args.dataset_name,
        args.split,
        args.output_path,
        args.sample_size,
        args.local_path,
        args.tokens_column,
        args.tags_column,
        args.text_column,
        args.annotations_column,
        args.inspect_only,
    )


if __name__ == "__main__":
    main()
