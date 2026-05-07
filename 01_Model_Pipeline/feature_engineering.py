"""Simple section-aware feature helpers.

The project mainly uses BERT token classification. This file is included because the
requested structure allows feature_engineering.py if needed. Here it supports the
section-aware approach with readable section summaries.
"""

from __future__ import annotations

from collections import Counter

from data_loading import ResumeRecord
from preprocessing import split_resume_into_sections


def count_detected_sections(records: list[ResumeRecord]) -> Counter:
    """Count how often each section type is detected."""

    counter = Counter()
    for record in records:
        for segment in split_resume_into_sections(record.content):
            counter[segment.section_name] += 1
    return counter


def print_section_summary(records: list[ResumeRecord]) -> None:
    """Print a small summary used to explain the proposed method."""

    counter = count_detected_sections(records)
    print("\nDetected resume sections:")
    for section_name, count in counter.most_common():
        print(f"- {section_name}: {count}")
