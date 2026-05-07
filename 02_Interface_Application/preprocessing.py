"""Preprocessing for BERT-based resume NER.

The document requires:
- minimal text cleaning
- BERT tokenization
- conversion from character-level labels to BIO labels
- handling long sequences
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from data_loading import Entity, ResumeRecord


SECTION_HEADERS = {
    "education": ["EDUCATION", "ACADEMIC BACKGROUND", "QUALIFICATION"],
    "experience": ["WORK EXPERIENCE", "EXPERIENCE", "EMPLOYMENT HISTORY"],
    "skills": ["SKILLS", "TECHNICAL SKILLS", "ADDITIONAL SKILLS"],
    "projects": ["PROJECTS", "PROJECT EXPERIENCE"],
    "language": ["LANGUAGES", "LANGUAGE"],
    "certifications": ["CERTIFICATIONS", "CERTIFICATES", "CERTIFICATES & GRANTS"],
}


@dataclass
class TextSegment:
    """A piece of resume text that will be sent to BERT."""

    text: str
    start_offset: int
    section_name: str


def clean_text_for_display(text: str) -> str:
    """Clean text lightly for printing output.

    Training keeps the original text because character labels depend on exact positions.
    """

    text = text.replace("â€¢", "-")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_label_list(records: list[ResumeRecord]) -> list[str]:
    """Create BIO labels from all entity types in the dataset."""

    entity_labels = sorted({entity.label for record in records for entity in record.entities})
    labels = ["O"]
    for label in entity_labels:
        labels.append(f"B-{label}")
        labels.append(f"I-{label}")
    return labels


def convert_entities_to_bio(
    text: str,
    entities: list[Entity],
    offsets: list[tuple[int, int]],
    label_to_id: dict[str, int],
    segment_start: int = 0,
) -> list[int]:
    """Convert character-level spans to token-level BIO labels.

    Each BERT token has a character offset. If the token overlaps an entity span,
    it receives B-Label for the first token and I-Label for the following tokens.
    """

    # Real text tokens start as O. Special tokens and padding have offset (0, 0),
    # so we mark them as -100 to make the loss and metrics ignore them.
    labels = [
        -100 if token_start == token_end else label_to_id["O"]
        for token_start, token_end in offsets
    ]
    sorted_entities = sorted(entities, key=lambda item: (item.start, item.end))

    for entity in sorted_entities:
        entity_start = entity.start - segment_start
        entity_end = entity.end - segment_start + 1
        first_entity_token = True

        for token_index, (token_start, token_end) in enumerate(offsets):
            if token_start == token_end:
                continue

            overlaps_entity = token_start < entity_end and token_end > entity_start
            if not overlaps_entity:
                continue

            prefix = "B" if first_entity_token else "I"
            labels[token_index] = label_to_id[f"{prefix}-{entity.label}"]
            first_entity_token = False

    return labels


def make_bert_features(
    records: list[ResumeRecord],
    tokenizer,
    label_to_id: dict[str, int],
    max_length: int = 256,
    section_aware: bool = False,
    ignore_overlapping_tokens: bool = True,
) -> list[dict]:
    """Tokenize resumes and create label IDs for training or evaluation."""

    features = []

    for record in records:
        segments = split_resume_into_sections(record.content) if section_aware else [
            TextSegment(text=record.content, start_offset=0, section_name="full_resume")
        ]

        for segment in segments:
            segment_entities = [
                entity
                for entity in record.entities
                if entity.start >= segment.start_offset
                and entity.end < segment.start_offset + len(segment.text)
            ]

            encoded_chunks = tokenizer(
                segment.text,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_offsets_mapping=True,
                return_overflowing_tokens=True,
                stride=64,
            )

            previous_chunk_end = -1
            for chunk_index in range(len(encoded_chunks["input_ids"])):
                offsets = encoded_chunks["offset_mapping"][chunk_index]
                labels = convert_entities_to_bio(
                    text=segment.text,
                    entities=segment_entities,
                    offsets=offsets,
                    label_to_id=label_to_id,
                    segment_start=segment.start_offset,
                )

                # Long resumes are split with stride, so two chunks can contain
                # the same text tokens. We keep overlap as context, but ignore it
                # in loss/metrics to avoid counting the same token twice.
                if ignore_overlapping_tokens and chunk_index > 0:
                    labels = [
                        -100 if token_end <= previous_chunk_end else label
                        for label, (token_start, token_end) in zip(labels, offsets)
                    ]

                real_token_ends = [
                    token_end
                    for token_start, token_end in offsets
                    if token_start != token_end
                ]
                if real_token_ends:
                    previous_chunk_end = max(previous_chunk_end, max(real_token_ends))

                features.append(
                    {
                        "input_ids": encoded_chunks["input_ids"][chunk_index],
                        "attention_mask": encoded_chunks["attention_mask"][chunk_index],
                        "labels": labels,
                        "section_name": segment.section_name,
                    }
                )

    return features


def split_resume_into_sections(text: str) -> list[TextSegment]:
    """Split a resume using simple section-header rules.

    This is the proposed section-aware step from the document. It is intentionally simple
    so it can be explained easily in a university presentation.
    """

    matches = []
    for section_name, headers in SECTION_HEADERS.items():
        for header in headers:
            pattern = rf"(?im)^\s*{re.escape(header)}\s*$"
            for match in re.finditer(pattern, text):
                matches.append((match.start(), section_name))

            spaced_pattern = _build_spaced_header_pattern(header)
            for match in re.finditer(spaced_pattern, text):
                matches.append((match.start(), section_name))

    # Some resume datasets store the whole resume in one long line, so headers
    # like WORK EXPERIENCE or SKILLS are not always alone on a line.
    if not matches:
        for section_name, headers in SECTION_HEADERS.items():
            for header in headers:
                pattern = rf"(?i)(?<![A-Za-z]){re.escape(header)}(?![A-Za-z])"
                for match in re.finditer(pattern, text):
                    matches.append((match.start(), section_name))

    if not matches:
        return [TextSegment(text=text, start_offset=0, section_name="unknown")]

    matches = sorted(set(matches))
    segments: list[TextSegment] = []

    if matches[0][0] > 0:
        segments.append(
            TextSegment(text=text[: matches[0][0]], start_offset=0, section_name="header")
        )

    for index, (start, section_name) in enumerate(matches):
        end = matches[index + 1][0] if index + 1 < len(matches) else len(text)
        segment_text = text[start:end]
        if segment_text.strip():
            segments.append(
                TextSegment(text=segment_text, start_offset=start, section_name=section_name)
            )

    return segments


def _build_spaced_header_pattern(header: str) -> str:
    """Allow headers like 'W O R K  E X P E R I E N C E'."""

    words = header.split()
    spaced_words = []
    for word in words:
        spaced_chars = r"\s*".join(re.escape(character) for character in word)
        spaced_words.append(spaced_chars)
    joined = r"\s+".join(spaced_words)
    return rf"(?im)^\s*{joined}\s*$"


def decode_entities_from_labels(
    tokens: list[str],
    label_ids: list[int],
    id_to_label: dict[int, str],
) -> dict[str, list[str]]:
    """Convert predicted BIO labels into a simple structured output dictionary."""

    extracted: dict[str, list[str]] = {}
    current_label = None
    current_tokens: list[str] = []

    for token, label_id in zip(tokens, label_ids):
        label = id_to_label.get(int(label_id), "O")

        if label == "O" or label_id == -100:
            if current_label and current_tokens:
                text = _join_wordpieces(current_tokens)
                extracted.setdefault(current_label, []).append(text)
            current_label = None
            current_tokens = []
            continue

        prefix, entity_label = label.split("-", 1)
        if prefix == "B" or entity_label != current_label:
            if current_label and current_tokens:
                text = _join_wordpieces(current_tokens)
                extracted.setdefault(current_label, []).append(text)
            current_label = entity_label
            current_tokens = [token]
        else:
            current_tokens.append(token)

    if current_label and current_tokens:
        text = _join_wordpieces(current_tokens)
        extracted.setdefault(current_label, []).append(text)

    return extracted


def _join_wordpieces(tokens: list[str]) -> str:
    """Turn BERT wordpieces back into readable text."""

    text = ""
    for token in tokens:
        if token in {"[CLS]", "[SEP]", "[PAD]"}:
            continue
        if token.startswith("##"):
            text += token[2:]
        else:
            text += " " + token
    return text.strip()
