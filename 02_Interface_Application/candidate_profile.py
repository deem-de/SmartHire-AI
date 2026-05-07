"""Helpers for connecting Part 1 resume parsing output to Part 2 interview flow.

This file keeps the connection simple: Part 1 produces a dictionary of extracted
entities, and Part 2 reads the same dictionary as a candidate profile.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


PROFILE_FIELDS = [
    "Name",
    "PERSON",
    "Email Address",
    "EMAIL",
    "PHONE",
    "Location",
    "LOCATION",
    "Skills",
    "SKILL",
    "SKILLS",
    "Languages",
    "LANGUAGE",
    "Degree",
    "EDUCATION",
    "College Name",
    "Graduation Year",
    "Companies worked at",
    "COMPANY",
    "ORG",
    "Designation",
    "DESIGNATION",
    "EXPERIENCE",
    "Years of Experience",
]

LABEL_ALIASES = {
    "PERSON": "Name",
    "EMAIL": "Email Address",
    "PHONE": "Phone",
    "LOCATION": "Location",
    "SKILL": "Skills",
    "SKILLS": "Skills",
    "LANGUAGE": "Languages",
    "EDUCATION": "Degree",
    "COMPANY": "Companies worked at",
    "ORG": "Companies worked at",
    "DESIGNATION": "Designation",
    "EXPERIENCE": "Designation",
    "EXPERTISE": "Skills",
    "CERTIFICATION": "Degree",
    "LANGUAGE": "Skills",
}


def build_profile_from_entities(entities: dict[str, list[str]]) -> dict[str, list[str]]:
    """Create a clean candidate profile from extracted resume entities."""

    profile: dict[str, list[str]] = {}

    for field_name in PROFILE_FIELDS:
        values = entities.get(field_name, [])
        output_field_name = LABEL_ALIASES.get(field_name, field_name)
        if field_name in {"Skills", "SKILL", "SKILLS", "EXPERTISE"}:
            cleaned_values = _clean_skill_values(values)
        elif output_field_name == "Languages":
            cleaned_values = _clean_language_values(values)
        elif output_field_name == "Name":
            cleaned_values = _clean_name_values(values)
        elif output_field_name == "Email Address":
            cleaned_values = _clean_email_values(values)
        elif output_field_name == "Designation":
            cleaned_values = _clean_designation_values(values)
        elif output_field_name == "Degree":
            cleaned_values = _clean_degree_values(values)
        else:
            cleaned_values = []

            for value in values:
                clean_value = _clean_general_value(str(value))
                if clean_value and clean_value not in cleaned_values:
                    cleaned_values.append(clean_value)

        if cleaned_values:
            existing_values = profile.setdefault(output_field_name, [])
            for clean_value in cleaned_values:
                if clean_value not in existing_values:
                    existing_values.append(clean_value)
            profile[output_field_name] = existing_values[:8]

    return profile


def _clean_name_values(values: list[str]) -> list[str]:
    """Keep likely person names and drop schools or other institutions."""

    blocked_words = {
        "college",
        "school",
        "university",
        "institute",
        "academy",
        "technology",
        "vidyalaya",
        "campus",
    }
    cleaned_names = []

    for value in values:
        clean_value = _clean_general_value(str(value))
        if not clean_value:
            continue

        words = clean_value.split()
        lower_words = {word.strip(".").lower() for word in words}
        if lower_words & blocked_words:
            continue
        if any(character.isdigit() for character in clean_value):
            continue
        if len(words) > 4:
            continue

        if clean_value not in cleaned_names:
            cleaned_names.append(clean_value)

    return cleaned_names[:2]


def _clean_email_values(values: list[str]) -> list[str]:
    """Normalize email-like values for cleaner display."""

    cleaned_emails = []
    for value in values:
        clean_value = _clean_general_value(str(value))
        if not clean_value:
            continue

        clean_value = re.sub(r"\s*([@./_-])\s*", r"\1", clean_value)
        clean_value = clean_value.replace("Indeed:", "").strip()
        if clean_value and clean_value not in cleaned_emails:
            cleaned_emails.append(clean_value)

    return cleaned_emails[:2]


def _clean_designation_values(values: list[str]) -> list[str]:
    """Drop date ranges and keep role-like text."""

    cleaned_designations = []
    month_pattern = re.compile(
        r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
        flags=re.IGNORECASE,
    )

    for value in values:
        clean_value = _clean_general_value(str(value))
        if not clean_value:
            continue
        if month_pattern.search(clean_value):
            continue
        if re.fullmatch(r"\d{4}", clean_value):
            continue
        if clean_value.lower().startswith("to "):
            continue

        if clean_value not in cleaned_designations:
            cleaned_designations.append(clean_value)

    return cleaned_designations[:4]


def _clean_degree_values(values: list[str]) -> list[str]:
    """Prefer degree-like text over year-only values."""

    cleaned_degrees = []
    year_only = []
    degree_keywords = {
        "b.e",
        "b.tech",
        "bachelor",
        "master",
        "m.e",
        "m.tech",
        "science",
        "engineering",
        "degree",
    }

    for value in values:
        clean_value = _clean_general_value(str(value))
        if not clean_value:
            continue

        lower_value = clean_value.lower()
        if re.fullmatch(r"\d{4}", clean_value):
            if clean_value not in year_only:
                year_only.append(clean_value)
            continue

        if any(keyword in lower_value for keyword in degree_keywords):
            if clean_value not in cleaned_degrees:
                cleaned_degrees.append(clean_value)

    if cleaned_degrees:
        return cleaned_degrees[:4]
    return year_only[:2]


def _clean_general_value(value: str) -> str:
    """Clean noisy model fragments from non-skill fields."""

    value = " ".join(value.split())
    value = value.strip(" .,:;-/")

    if not value:
        return ""
    if len(value) <= 1:
        return ""
    if value.lower() in {"sk", "ls", "ation", "b"}:
        return ""

    return value


def _clean_skill_values(values: list[str]) -> list[str]:
    """Split long skills sections into short readable skill names."""

    cleaned_skills = []

    for value in values:
        text = str(value).replace("•", ",").replace("â€¢", ",")
        text = re.sub(r"\bNon\s*-\s*Technical Skills\b.*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\([^)]*\)", "", text)
        parts = re.split(r"[,;\n]+", text)

        for part in parts:
            skill = " ".join(part.split())
            if ":" in skill:
                skill = skill.split(":", 1)[1].strip()

            skill = skill.strip(" .:-")
            if not skill:
                continue

            lower_skill = skill.lower()
            if lower_skill in {
                "technical skills",
                "non - technical skills",
                "additional information",
                "programming language",
                "programming languages",
                "sk",
                "ls",
            }:
                continue

            if len(skill) <= 1:
                continue

            if len(skill) > 40:
                continue

            if skill not in cleaned_skills:
                cleaned_skills.append(skill)

    return cleaned_skills[:12]


def _clean_language_values(values: list[str]) -> list[str]:
    allowed = {
        "english",
        "arabic",
        "french",
        "spanish",
        "german",
        "urdu",
        "hindi",
        "turkish",
        "chinese",
        "japanese",
        "korean",
    }
    cleaned_languages = []
    for value in values:
        cleaned = _clean_general_value(str(value))
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered in allowed and cleaned.title() not in cleaned_languages:
            cleaned_languages.append(cleaned.title())
    return cleaned_languages[:5]


def save_profile(profile: dict, output_path: str | Path) -> None:
    """Save a candidate profile as JSON so Part 2 can read it."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(profile, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_profile(profile_path: str | Path) -> dict:
    """Load a candidate profile JSON file."""

    return json.loads(Path(profile_path).read_text(encoding="utf-8"))
