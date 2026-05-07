"""Final SmartHire application server.

This server connects the React demo UI to the real resume parsing pipeline:
- upload PDF/DOCX/TXT
- extract raw text
- run the trained BERT parser
- build a candidate profile
- generate interview questions
- evaluate interview answers

It also serves the React UI from the same app so the project can be demoed as a
single product instead of separate terminal commands.
"""

from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import torch
from transformers import AutoModelForTokenClassification

from candidate_profile import build_profile_from_entities
from document_loader import extract_text_from_document
from export_profile_from_model import predict_entities
from interview_evaluator import create_final_summary, evaluate_answer
from interview_fallback import (
    create_template_summary,
    evaluate_template_answer,
    generate_template_questions,
)
from interview_generator import generate_questions
from models import get_device, load_tokenizer_for_model
from preprocessing import SECTION_HEADERS, TextSegment, decode_entities_from_labels, split_resume_into_sections


ROOT_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = ROOT_DIR / "outputs"
FRONTEND_DIR = ROOT_DIR / "react-ui"
UPLOADS_DIR = OUTPUTS_DIR / "temp_uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = get_device()
MAX_LENGTH = 128
PARSER_MODEL_PATHS = {
    "baseline": OUTPUTS_DIR / "baseline_model",
    "section_aware": OUTPUTS_DIR / "section_aware_model",
}

MODEL_CACHE: dict[str, dict[str, Any]] = {}
MODEL_LOCK = threading.Lock()


class InterviewStartRequest(BaseModel):
    profile: dict[str, list[str]]
    language: str = "English"
    selected_model: str = "BERT + Gemini"
    question_count: int = Field(default=3, ge=1, le=5)


class EvaluateAnswerRequest(BaseModel):
    question: dict[str, Any]
    answer: str
    profile: dict[str, list[str]]
    language: str = "English"
    selected_model: str = "BERT + Gemini"


class FinalSummaryRequest(BaseModel):
    results: list[dict[str, Any]]
    language: str = "English"
    selected_model: str = "BERT + Gemini"


app = FastAPI(title="SmartHire API", version="1.0.0")


@app.middleware("http")
async def disable_frontend_cache(request, call_next):
    response = await call_next(request)
    if request.method == "GET" and not request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_metrics() -> dict[str, Any]:
    metrics_path = OUTPUTS_DIR / "metrics.json"
    if not metrics_path.exists():
        return {}
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def get_best_parser_key(metrics: dict[str, Any]) -> str:
    baseline_f1 = metrics.get("baseline", {}).get("f1", 0.0)
    section_f1 = metrics.get("section_aware", {}).get("f1", 0.0)
    return "section_aware" if section_f1 >= baseline_f1 else "baseline"


def get_id_to_label(model) -> dict[int, str]:
    """Read BIO labels from the saved model config."""

    id_to_label = {}
    for key, value in model.config.id2label.items():
        id_to_label[int(key)] = value
    return id_to_label


METRICS = load_metrics()
BEST_PARSER_KEY = get_best_parser_key(METRICS)


def get_parser_bundle(parser_key: str) -> dict[str, Any]:
    key = parser_key if parser_key in PARSER_MODEL_PATHS else BEST_PARSER_KEY

    with MODEL_LOCK:
        if key in MODEL_CACHE:
            return MODEL_CACHE[key]

        model_path = PARSER_MODEL_PATHS[key]
        tokenizer = load_tokenizer_for_model(str(model_path))
        model = AutoModelForTokenClassification.from_pretrained(str(model_path)).to(DEVICE)

        bundle = {
            "key": key,
            "path": str(model_path),
            "tokenizer": tokenizer,
            "model": model,
            "section_aware": key == "section_aware",
        }
        MODEL_CACHE[key] = bundle
        return bundle


def save_upload(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "resume.txt").suffix or ".txt"
    safe_name = Path(upload.filename or f"upload{suffix}").name
    output_path = UPLOADS_DIR / safe_name

    content = upload.file.read()
    output_path.write_bytes(content)
    return output_path


def flatten_first(profile: dict[str, list[str]], field_name: str, default: str = "Not detected") -> str:
    values = profile.get(field_name, [])
    return values[0] if values else default


def extract_text_fallback_profile(raw_text: str) -> dict[str, list[str]]:
    """Read obvious fields directly from the extracted text for cleaner display.

    This does not replace the trained BERT pipeline. It only fills obvious
    missing fields for the final UI when the NER output is noisy.
    """

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    fallback: dict[str, list[str]] = {}
    sections = collect_section_lines(raw_text)
    header_lines = sections.get("header", lines[:8])
    education_lines = sections.get("education", []) or get_section_lines(lines, "education")
    experience_lines = sections.get("experience", []) or get_section_lines(lines, "experience", "work experience")
    skills_lines = sections.get("skills", []) or get_section_lines(lines, "skills", "technical skills", "technical competencies")
    language_lines = sections.get("language", []) or get_section_lines(lines, "language", "languages")
    project_lines = sections.get("projects", []) or get_section_lines(lines, "projects", "project")

    name_candidate = find_name_candidate(header_lines)
    if name_candidate:
        fallback["Name"] = [name_candidate]

    title_candidate = find_title_candidate(header_lines)
    if title_candidate:
        fallback["Designation"] = [title_candidate]

    email_candidate = find_email_candidate(raw_text)
    if email_candidate:
        fallback["Email Address"] = [email_candidate]

    phone_candidate = find_phone_candidate(raw_text)
    if phone_candidate:
        fallback["Phone"] = [phone_candidate]

    location_candidate = find_location_candidate(header_lines) or find_location_candidate(lines)
    if location_candidate:
        fallback["Location"] = [location_candidate]

    experience_title = find_experience_title(experience_lines)
    experience_company = find_experience_company(experience_lines)
    experience_highlights = find_experience_highlights(experience_lines)

    if experience_title:
        fallback["Designation"] = [experience_title]
    if experience_company:
        fallback["Companies worked at"] = [experience_company]
    if experience_highlights:
        fallback["Experience Highlights"] = experience_highlights

    degree_candidate, school_candidate, year_candidate = find_education_details(
        education_lines,
        lines,
    )
    if degree_candidate:
        fallback["Degree"] = [degree_candidate]
    if school_candidate:
        fallback["College Name"] = [school_candidate]
    if year_candidate:
        fallback["Graduation Year"] = [year_candidate]

    skills = find_skill_candidates(skills_lines)
    text_level_skills = extract_skill_candidates_from_text(raw_text)
    if text_level_skills:
        skills = dedupe_keep_order(text_level_skills + skills)
    skills = refine_skill_values(skills)
    if skills:
        fallback["Skills"] = skills

    languages = find_language_candidates(language_lines, lines)
    if languages:
        fallback["Languages"] = languages[:6]

    project_candidate = find_project_candidate(project_lines)
    if project_candidate:
        fallback["Projects"] = [project_candidate]

    about_summary = find_about_summary(lines)
    if about_summary:
        fallback["About"] = [about_summary]

    return fallback


def get_section_lines(lines: list[str], *section_titles: str) -> list[str]:
    normalized_targets = {title.strip().lower() for title in section_titles}
    for index, line in enumerate(lines):
        if line.strip().lower() in normalized_targets:
            collected = []
            for next_line in lines[index + 1 :]:
                if next_line.isupper() and len(next_line.split()) <= 4:
                    break
                collected.append(next_line)
            return collected
    return []


def collect_section_lines(raw_text: str) -> dict[str, list[str]]:
    section_lines: dict[str, list[str]] = {}
    for segment in split_resume_into_sections(raw_text):
        lines = [clean_resume_line(line) for line in segment.text.splitlines() if line.strip()]
        if segment.section_name in SECTION_HEADERS and lines and line_is_section_header(lines[0], segment.section_name):
            lines = lines[1:]
        if lines:
            section_lines.setdefault(segment.section_name, []).extend(lines)
    return section_lines


def line_is_section_header(line: str, section_name: str) -> bool:
    normalized = re.sub(r"\s+", " ", line.upper()).strip()
    known_headers = {header.upper() for header in SECTION_HEADERS.get(section_name, [])}
    return normalized in known_headers


def clean_resume_line(line: str) -> str:
    line = line.replace("\u2022", " ").replace("\u2014", " - ").replace("\u2013", " - ")
    line = re.sub(r"\s{2,}", " ", line)
    return line.strip(" -|•\t")


def normalize_name_case(text: str) -> str:
    cleaned = clean_resume_line(text)
    if cleaned.isupper():
        return cleaned.title()
    return cleaned


def strip_date_spans(line: str, remove_years: bool = False) -> str:
    cleaned = clean_resume_line(line)
    month_names = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    cleaned = re.sub(
        rf"(?i)\b{month_names}\s+\d{{4}}\s*(?:-|to)\s*(?:present|{month_names}\s+\d{{4}})\b",
        "",
        cleaned,
    )
    cleaned = re.sub(r"(?i)\b\d{4}\s*(?:-|to)\s*(?:present|\d{4})\b", "", cleaned)
    if remove_years:
        cleaned = re.sub(r"\b(19|20)\d{2}\b", "", cleaned)
    return re.sub(r"\s{2,}", " ", cleaned).strip(" -,")


def looks_like_degree(text: str) -> bool:
    lowered = text.lower()
    degree_keywords = {
        "bachelor",
        "master",
        "phd",
        "b.sc",
        "b.s.",
        "b.e",
        "btech",
        "b.tech",
        "b. a.",
        "ba ",
        "mba",
        "m. a.",
        "m.a.",
        "b.a.",
        "diploma",
        "information systems",
        "computer science",
        "engineering",
        "commerce",
    }
    return any(keyword in lowered for keyword in degree_keywords)


def looks_like_location(text: str) -> bool:
    cleaned = clean_resume_line(text)
    lowered = cleaned.lower()
    if "@" in cleaned or re.search(r"\b(?:phone|email)\b", cleaned, flags=re.I):
        return False
    if looks_like_name(cleaned):
        return False
    if looks_like_title(cleaned) or looks_like_organization(cleaned):
        return False
    if any(token in lowered for token in ["lorem", "ipsum", "finallygot", "goddamn", "about me"]):
        return False
    if any(len(token) > 20 for token in cleaned.split()):
        return False
    if re.search(r"\b(?:st\.?|street|road|rd\.?|avenue|ave\.?|city|district|block|building|apartment|apt\.?|minsk)\b", lowered):
        return True
    if cleaned.count(",") >= 1 and 2 <= len(cleaned.split()) <= 12:
        return True
    if 1 <= len(cleaned.split()) <= 4 and cleaned.istitle() and re.fullmatch(r"[A-Za-z\s.-]+", cleaned):
        return True
    return False


def find_location_candidate(lines: list[str]) -> str | None:
    label_keywords = {"address", "adres", "location", "city"}
    for index, line in enumerate(lines[:30]):
        cleaned = clean_resume_line(line)
        lowered = cleaned.lower().rstrip(":")
        if lowered in label_keywords and index + 1 < len(lines):
            next_line = clean_resume_line(lines[index + 1])
            if next_line and looks_like_location(next_line):
                return next_line

        inline_match = re.match(r"(?i)^(?:address|adres|location|city)\s*[:\-]?\s*(.+)$", cleaned)
        if inline_match:
            candidate = clean_resume_line(inline_match.group(1))
            if candidate and looks_like_location(candidate):
                return candidate

    for line in lines[:20]:
        cleaned = clean_resume_line(line)
        if not cleaned:
            continue
        if looks_like_location(cleaned):
            return cleaned
    return None


def looks_like_skill_candidate(text: str) -> bool:
    cleaned = clean_resume_line(text)
    lowered = cleaned.lower()
    if not cleaned or "@" in cleaned or re.search(r"\d{4}", cleaned):
        return False
    if any(phrase in lowered for phrase in ["coding languages", "coding frameworks", "coding databases"]):
        return False
    if looks_like_organization(cleaned) or looks_like_title(cleaned) or looks_like_degree(cleaned):
        return False
    if any(word in lowered for word in ["visited ", "built relationships", "meet with clients", "train junior", "provided ", "assisted "]):
        return False
    return len(cleaned.split()) <= 5 or "," in cleaned


def dedupe_keep_order(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = value.lower().strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            output.append(value)
    return output


def filter_skill_values(values: list[str]) -> list[str]:
    short_skill_whitelist = {"sql", "api", "ui", "ux", "ml", "ai", "nlp", "aws", "gcp", "etl", "erp", "crm", "erp", "seo", "hr"}
    acronym_skill_whitelist = {"SQL", "API", "UI", "UX", "ML", "AI", "NLP", "AWS", "GCP", "ETL", "ERP", "CRM", "BI", "IT", "C", "C++"}
    canonical_single_word_map = {
        "python": "Python",
        "javascript": "JavaScript",
        "html": "HTML",
        "css": "CSS",
        "latex": "LaTeX",
        "java": "Java",
        "excel": "Excel",
        "tableau": "Tableau",
        "statistics": "Statistics",
        "fastapi": "FastAPI",
        "pytorch": "PyTorch",
        "tensorflow": "TensorFlow",
        "sales": "Sales",
        "retail": "Retail",
        "leadership": "Leadership",
        "communication": "Communication",
        "negotiation": "Negotiation",
        "linux": "Linux",
        "windows": "Windows",
        "oracle": "Oracle",
        "peoplesoft": "PeopleSoft",
        "iot": "IoT",
        "networking": "Networking",
        "research": "Research",
        "algorithms": "Algorithms",
    }
    single_word_skill_whitelist = {
        "python",
        "javascript",
        "html",
        "css",
        "latex",
        "java",
        "excel",
        "tableau",
        "statistics",
        "fastapi",
        "pytorch",
        "tensorflow",
        "sales",
        "retail",
        "leadership",
        "communication",
        "negotiation",
        "linux",
        "windows",
        "oracle",
        "peoplesoft",
        "iot",
        "networking",
        "research",
        "algorithms",
    }
    suspicious_exact = {
        "com",
        "email",
        "phone",
        "education",
        "work experience",
        "experience",
        "skills",
        "language",
        "references",
        "analysis",
        "data",
        "professional",
        "relationships",
        "customers",
        "point of contact",
        "outreach",
        "purchase",
        "new opportunities",
        "bachelor of",
        "science",
        "upe",
        "state",
        "plans",
        "lessons",
        "instruction",
        "biology",
        "education",
        "references",
        "language",
        "project",
        "projects",
        "dataset",
        "datasets",
        "insight",
        "insights",
        "ducation",
        "management",
        "start",
        "black",
        "sparrow",
        "programming",
        "degrees",
        "curriculum",
        "certified",
        "personal",
        "interests",
        "talks",
        "publications",
        "grants",
        "aboutme",
        "about",
        "captain",
    }
    filtered = []
    for value in values:
        cleaned = clean_resume_line(value)
        lowered = cleaned.lower()
        if not cleaned or lowered in suspicious_exact:
            continue
        if lowered.startswith(("preparation in ", "understanding the ", "knowledge in ", "experience with ")):
            continue
        if lowered.startswith("and "):
            continue
        if looks_like_action_sentence(cleaned) or looks_like_organization(cleaned):
            continue
        if looks_like_title(cleaned):
            continue
        if len(cleaned.split()) == 1:
            if lowered in single_word_skill_whitelist:
                filtered.append(canonical_single_word_map.get(lowered, cleaned.title() if lowered not in short_skill_whitelist else cleaned.upper()))
                continue
            if cleaned in acronym_skill_whitelist:
                filtered.append(cleaned)
                continue
            if len(cleaned) <= 3 and lowered not in short_skill_whitelist:
                continue
            # Reject generic single words unless they are known skill-like tokens.
            continue
        if len(cleaned.split()) == 1 and len(cleaned) <= 3 and lowered not in short_skill_whitelist:
            continue
        filtered.append(cleaned)
    return dedupe_keep_order(filtered)


def refine_skill_values(values: list[str]) -> list[str]:
    canonical_forms = {canonical.lower(): canonical for _, canonical in KNOWN_SKILL_PATTERNS}
    canonical_patterns = [(pattern.lower(), canonical) for pattern, canonical in KNOWN_SKILL_PATTERNS]

    normalized: list[str] = []
    for value in values:
        cleaned = clean_resume_line(value).strip(" .,:;()")
        lowered = cleaned.lower()
        if lowered in canonical_forms:
            normalized.append(canonical_forms[lowered])
            continue

        overlapping = [canonical for pattern, canonical in canonical_patterns if pattern in lowered]
        if overlapping and (len(cleaned.split()) > 4 or any(ch in value for ch in "().")):
            continue

        normalized.append(cleaned)

    refined = dedupe_keep_order(filter_skill_values(normalized))
    if "STL Containers" in refined:
        refined = [value for value in refined if value != "STL"]
    if "Mathematical Statistics" in refined:
        refined = [value for value in refined if value != "Statistics"]
    return refined


def score_skill_list(values: list[str]) -> float:
    if not values:
        return 0.0

    score = 0.0
    for value in values:
        cleaned = clean_resume_line(value)
        lowered = cleaned.lower()
        if not cleaned:
            continue
        if len(cleaned.split()) >= 2:
            score += 2.5
        elif lowered in {"python", "java", "excel", "tableau", "statistics", "sales", "retail", "fastapi"}:
            score += 2.0
        elif cleaned in {"SQL", "API", "UI", "UX", "ML", "AI", "NLP", "AWS", "GCP", "ETL", "ERP", "CRM", "BI", "IT", "C", "C++"}:
            score += 1.5
        else:
            score -= 1.0
    return score / max(len(values), 1)


def find_name_candidate(lines: list[str]) -> str | None:
    for line in lines[:8]:
        candidate = normalize_name_case(line)
        if looks_like_name(candidate):
            return candidate
    return None


def find_title_candidate(lines: list[str]) -> str | None:
    for line in lines[:8]:
        candidate = strip_date_spans(line)
        relationship_match = re.search(r"(?i)^(.+?)\s+of\s+the\s+.+$", candidate)
        if relationship_match:
            candidate = clean_resume_line(relationship_match.group(1))
        candidate = re.sub(r"(?i)\bcodewars\b.*$", "", candidate).strip(" -,")
        if looks_like_title(candidate) and not looks_like_organization(candidate):
            return candidate
    return None


def find_experience_title(lines: list[str]) -> str | None:
    title_candidates: list[str] = []
    for line in lines:
        candidate = strip_date_spans(line)
        relationship_match = re.search(r"(?i)^(.+?)\s+of\s+the\s+.+$", candidate)
        if relationship_match:
            candidate = clean_resume_line(relationship_match.group(1))
        if (
            not candidate
            or looks_like_degree(candidate)
            or looks_like_location(candidate)
            or looks_like_organization(candidate)
        ):
            continue
        if looks_like_title(candidate):
            title_candidates.append(candidate)
    return dedupe_keep_order(title_candidates)[0] if title_candidates else None


def find_experience_company(lines: list[str]) -> str | None:
    if not lines:
        return None
    cleaned_lines = [strip_date_spans(line) for line in lines if strip_date_spans(line)]

    for candidate in cleaned_lines:
        relationship_match = re.search(r"(?i)\b(?:captain|manager|engineer|analyst|developer|specialist|representative|associate|consultant|intern)\s+of\s+the\s+(.+)$", candidate)
        if relationship_match:
            organization = clean_resume_line(relationship_match.group(1))
            if organization:
                return organization

    title_indexes = [
        index
        for index, candidate in enumerate(cleaned_lines)
        if looks_like_title(candidate) and not looks_like_degree(candidate) and not looks_like_organization(candidate)
    ]

    company_candidates: list[str] = []
    for index, candidate in enumerate(cleaned_lines):
        if not candidate or looks_like_degree(candidate):
            continue
        if looks_like_organization(candidate):
            company_candidates.append(candidate)
            if title_indexes and any(abs(index - title_index) <= 2 for title_index in title_indexes):
                if "university" not in candidate.lower() and "college" not in candidate.lower():
                    return candidate

    filtered = [
        candidate
        for candidate in dedupe_keep_order(company_candidates)
        if "university" not in candidate.lower() and "college" not in candidate.lower()
    ]
    if filtered:
        return filtered[0]
    return dedupe_keep_order(company_candidates)[0] if company_candidates else None


def find_education_details(education_lines: list[str], all_lines: list[str]) -> tuple[str | None, str | None, str | None]:
    candidate_lines = education_lines or [
        line
        for line in all_lines
        if looks_like_degree(line)
        or any(token in line.lower() for token in ["university", "college", "school", "institute", " uni"])
    ]
    degree = None
    school = None
    year = None

    for index, line in enumerate(candidate_lines):
        cleaned = clean_resume_line(line)
        if not degree and looks_like_degree(cleaned):
            degree = strip_date_spans(cleaned, remove_years=True)
            if index + 1 < len(candidate_lines):
                next_line = clean_resume_line(candidate_lines[index + 1])
                if (
                    next_line
                    and not re.search(r"\b(?:cgpa|gpa)\b", next_line, flags=re.I)
                    and not re.search(r"\b((?:19|20)\d{2})\b", next_line)
                    and not looks_like_organization(next_line)
                    and len(next_line.split()) <= 8
                ):
                    degree = f"{degree} {next_line}".strip()
        if not school and any(token in cleaned.lower() for token in ["university", "college", "school", "institute", " uni"]):
            school = cleaned
        if not year:
            range_match = re.search(r"\b((?:19|20)\d{2})\s*(?:-|–|—|to)\s*((?:19|20)\d{2})\b", cleaned, flags=re.I)
            if range_match:
                year = range_match.group(2)
            else:
                year_match = re.search(r"\b(19|20)\d{2}\b", cleaned)
                if year_match:
                    year = year_match.group(0)

        if degree and school and year:
            break

    if not year and (degree or school):
        relevant_indexes = [
            index
            for index, line in enumerate(all_lines)
            if (degree and clean_resume_line(line) == degree) or (school and clean_resume_line(line) == school)
        ]
        for index in relevant_indexes:
            window = all_lines[max(0, index - 2) : index + 3]
            for nearby_line in window:
                cleaned_nearby = clean_resume_line(nearby_line)
                range_match = re.search(r"\b(19|20)\d{2}\s*[-–to]+\s*((19|20)\d{2})\b", cleaned_nearby, flags=re.I)
                if range_match:
                    year = range_match.group(2)
                    break
                year_match = re.search(r"\b(19|20)\d{2}\b", cleaned_nearby)
                if year_match:
                    year = year_match.group(0)
                    break
            if year:
                break

    return degree, school, year


def find_skill_candidates(lines: list[str]) -> list[str]:
    skills: list[str] = []
    for line in lines:
        cleaned = clean_resume_line(line)
        if not cleaned:
            continue
        if cleaned.lower().startswith("other "):
            cleaned = cleaned[6:].strip()
        normalized_header = cleaned.upper().strip()
        if normalized_header in {"LANGUAGE", "REFERENCES", "LANGUAGE REFERENCES"}:
            break
        if any(token in cleaned.upper() for token in ["PHONE", "EMAIL"]):
            continue
        if re.match(r"(?i)^(common|features|technical skills|technical competencies|c\+\+|python|java|javascript|sql)\s*$", cleaned):
            continue
        if "," in cleaned:
            parts = [clean_resume_line(part) for part in cleaned.split(",")]
        else:
            parts = [cleaned]
        for part in parts:
            if looks_like_skill_candidate(part):
                skills.append(part)
    return filter_skill_values(skills)


KNOWN_SKILL_PATTERNS: list[tuple[str, str]] = [
    ("linear algebra", "Linear Algebra"),
    ("discrete mathematics", "Discrete Mathematics"),
    ("mathematical analysis", "Mathematical Analysis"),
    ("probability theory", "Probability Theory"),
    ("mathematical statistics", "Mathematical Statistics"),
    ("object oriented programming", "Object-Oriented Programming"),
    ("oop", "Object-Oriented Programming"),
    ("algorithms", "Algorithms"),
    ("data structures", "Data Structures"),
    ("structure data", "Data Structures"),
    ("c++", "C++"),
    ("stl containers", "STL Containers"),
    ("stl", "STL"),
    ("power bi", "Power BI"),
    ("data analysis", "Data Analysis"),
    ("dashboard design", "Dashboard Design"),
    ("statistics", "Statistics"),
    ("python", "Python"),
    ("sql", "SQL"),
    ("excel", "Excel"),
    ("html", "HTML"),
    ("css", "CSS"),
    ("javascript", "JavaScript"),
    ("latex", "LaTeX"),
    ("linux", "Linux"),
    ("android", "Android"),
    ("user research", "User Research"),
    ("people management", "People Management"),
    ("fast-moving consumer goods", "Fast-moving Consumer Goods"),
    ("packaged consumer goods", "Packaged Consumer Goods"),
    ("corporate sales account", "Corporate Sales Account"),
    ("sales", "Sales"),
    ("retail", "Retail"),
    ("privateering", "Privateering"),
    ("bucaneering", "Bucaneering"),
    ("rum", "Rum"),
    ("parler", "Parler"),
]


def extract_skill_candidates_from_text(raw_text: str) -> list[str]:
    normalized = clean_resume_line(raw_text).lower()
    found: list[str] = []
    for pattern, canonical in KNOWN_SKILL_PATTERNS:
        if re.search(rf"(?<![A-Za-z]){re.escape(pattern)}(?![A-Za-z])", normalized, flags=re.I):
            found.append(canonical)
    return dedupe_keep_order(found)


def find_project_candidate(lines: list[str]) -> str | None:
    for line in lines:
        cleaned = strip_date_spans(line)
        if cleaned and "@" not in cleaned and not looks_like_action_sentence(cleaned):
            return cleaned
    return None


def find_language_candidates(section_lines: list[str], all_lines: list[str]) -> list[str]:
    source_lines = section_lines or get_section_lines(all_lines, "language", "languages")

    known_languages = {
        "russian",
        "english",
        "arabic",
        "french",
        "german",
        "spanish",
        "urdu",
        "hindi",
        "turkish",
        "italian",
        "chinese",
        "mandarin",
        "korean",
        "japanese",
    }
    collected: list[str] = []
    for line in (source_lines or all_lines):
        cleaned = clean_resume_line(line)
        if not cleaned or any(token in cleaned.lower() for token in ["phone", "email", "@"]):
            continue
        inline_match = re.match(r"(?i)^languages?\s*[:\-]?\s*(.+)$", cleaned)
        if inline_match:
            cleaned = clean_resume_line(inline_match.group(1))
        if "," in cleaned:
            parts = [clean_resume_line(part) for part in cleaned.split(",")]
        else:
            parts = re.split(r"\s{2,}|/|\|", cleaned)
            parts = [clean_resume_line(part) for part in parts if clean_resume_line(part)]
            if len(parts) == 1:
                parts = [cleaned]
        for part in parts:
            lowered = part.lower()
            if lowered in known_languages:
                collected.append(part.title())
    return dedupe_keep_order(collected)


def find_about_summary(lines: list[str]) -> str | None:
    for index, line in enumerate(lines):
        normalized = clean_resume_line(line).lower().replace(" ", "")
        if normalized in {"aboutme", "about"}:
            collected: list[str] = []
            for next_line in lines[index + 1 : index + 6]:
                cleaned = clean_resume_line(next_line)
                if not cleaned or len(cleaned.split()) < 5:
                    continue
                if "@" in cleaned or re.search(r"\+?\d[\d\s()/.-]{7,}\d", cleaned):
                    continue
                if looks_like_location(cleaned):
                    continue
                if any(token in cleaned.lower() for token in ["education", "experience", "skills", "languages", "certificates", "publications", "talks"]):
                    break
                collected.append(cleaned)
            if collected:
                return " ".join(collected[:2])[:320].strip()
    return None


def find_experience_highlights(lines: list[str]) -> list[str]:
    highlights: list[str] = []
    for line in lines:
        cleaned = clean_resume_line(line)
        if not cleaned or "@" in cleaned:
            continue
        if looks_like_action_sentence(cleaned) and len(cleaned.split()) >= 4:
            highlights.append(cleaned)
    return dedupe_keep_order(highlights)[:4]


def derive_major_from_degree(degree: str) -> str:
    if not degree or degree == "Not detected":
        return "Not detected"
    cleaned = clean_resume_line(degree)
    cleaned = cleaned.replace("·", " · ")
    parts = [part.strip() for part in cleaned.split("·") if part.strip()]
    if parts and re.fullmatch(r"[A-Za-z.\s]+", parts[0]) and len(parts[0].split()) <= 3:
        cleaned = parts[0]
    cleaned = re.sub(r"^(Bachelor|Master|B\.?A\.?|B\.?Sc\.?|B\.?E\.?|MBA|M\.?Sc\.?)\s+(of\s+)?", "", cleaned, flags=re.I)
    cleaned = re.sub(r"^(Bachelor of|Master of)\s+", "", cleaned, flags=re.I)
    cleaned = cleaned.strip(" -,:")
    if not cleaned or re.fullmatch(r"[BM]\.?\s*A\.?", cleaned, flags=re.I):
        return "Not detected"
    return cleaned


def normalize_degree_display(degree: str) -> str:
    if not degree or degree == "Not detected":
        return "Not detected"
    cleaned = clean_resume_line(degree).replace("·", " · ")
    parts = [part.strip() for part in cleaned.split("·") if part.strip()]
    if parts:
        return parts[0]
    return cleaned


def normalize_university_display(university: str) -> str:
    if not university or university == "Not detected":
        return "Not detected"
    cleaned = clean_resume_line(university).replace("·", " · ")
    parts = [part.strip() for part in cleaned.split("·") if part.strip()]
    if len(parts) >= 2:
        return parts[-1]
    return cleaned


def ensure_sentence(text: str) -> str:
    cleaned = clean_resume_line(text)
    if not cleaned:
        return cleaned
    if cleaned[-1] not in ".!?":
        return f"{cleaned}."
    return cleaned


def remove_context_tokens_from_skills(skills: list[str], *context_values: str) -> list[str]:
    blocked_tokens: set[str] = set()
    for value in context_values:
        if not value or value == "Not detected":
            continue
        for token in re.findall(r"[A-Za-z][A-Za-z+-]*", value.lower()):
            if len(token) >= 4:
                blocked_tokens.add(token)

    cleaned_skills: list[str] = []
    for skill in skills:
        normalized = skill.lower().strip()
        tokens = re.findall(r"[A-Za-z][A-Za-z+-]*", normalized)
        relevant_tokens = [token for token in tokens if len(token) >= 4]
        if relevant_tokens and all(token in blocked_tokens for token in relevant_tokens):
            continue
        cleaned_skills.append(skill)
    return dedupe_keep_order(cleaned_skills)


def looks_like_name(text: str) -> bool:
    words = text.split()
    if not 2 <= len(words) <= 4:
        return False
    if any(any(char.isdigit() for char in word) for word in words):
        return False
    blocked = {"skills", "experience", "education", "email", "resume", "project", "about", "interview", "analysis"}
    if any(word.lower() in blocked for word in words):
        return False
    if looks_like_title(text) or looks_like_organization(text):
        return False
    return all(re.fullmatch(r"[A-Za-z][A-Za-z'.-]*", word) for word in words)


def looks_like_organization(text: str) -> bool:
    keywords = {
        "company",
        "inc",
        "corp",
        "corporation",
        "industries",
        "industry",
        "ltd",
        "llc",
        "technologies",
        "technology",
        "tech",
        "university",
        "institute",
        "college",
        "school",
        "lab",
        "analytics",
        "solutions",
        "systems",
        "group",
    }
    words = {word.lower().strip(".,") for word in text.split()}
    return bool(words & keywords)


def looks_like_title(text: str) -> bool:
    keywords = {
        "intern",
        "engineer",
        "developer",
        "analyst",
        "scientist",
        "associate",
        "representative",
        "seller",
        "agent",
        "support",
        "manager",
        "specialist",
        "consultant",
        "administrator",
        "student",
        "trainee",
        "captain",
        "pirate",
    }
    words = {word.lower().strip(".,") for word in text.split()}
    return bool(words & keywords)


def looks_like_valid_email(text: str) -> bool:
    return bool(re.fullmatch(r"[\w.+-]+@[\w.-]+\.\w+", text.strip()))


def looks_like_valid_phone(text: str) -> bool:
    digits_only = re.sub(r"\D", "", text)
    return len(digits_only) >= 8


def find_email_candidate(raw_text: str) -> str | None:
    matches = re.findall(r"[\w.+-]+@[\w.-]+\.\w+", raw_text)
    cleaned = [match.strip() for match in matches if looks_like_valid_email(match.strip())]
    return cleaned[0] if cleaned else None


def find_phone_candidate(raw_text: str) -> str | None:
    candidates = re.findall(r"(\(?\+?\d[\d\s()/.-]{7,}\d)", raw_text)
    cleaned_candidates: list[str] = []
    for candidate in candidates:
        cleaned = clean_resume_line(candidate)
        digits = re.sub(r"\D", "", cleaned)
        if len(digits) >= 8:
            cleaned_candidates.append(normalize_phone_display(cleaned))
    if not cleaned_candidates:
        return None
    cleaned_candidates.sort(key=lambda item: len(re.sub(r"\D", "", item)), reverse=True)
    return cleaned_candidates[0]


def normalize_phone_display(text: str) -> str:
    cleaned = clean_resume_line(text)
    digits = re.sub(r"\D", "", cleaned)
    if len(digits) < 8:
        return cleaned

    if cleaned.startswith("+") or "(+" in text:
        if len(digits) > 9:
            country = digits[:-9]
            local = digits[-9:]
            return f"+{country} {local[:2]} {local[2:5]} {local[5:]}"
        return f"+{digits}"

    return digits


def looks_like_action_sentence(text: str) -> bool:
    action_words = {
        "built",
        "created",
        "developed",
        "found",
        "improved",
        "maintained",
        "provided",
        "managed",
        "analyzed",
        "designed",
        "assisted",
        "offer",
        "offered",
        "meet",
        "met",
        "train",
        "trained",
        "visit",
        "visited",
        "documented",
        "cleaned",
        "presented",
        "renew",
        "updated",
    }
    words = {word.lower().strip(".,") for word in text.split()}
    return bool(words & action_words)


def merge_profile_with_fallback(profile: dict[str, list[str]], raw_text: str) -> dict[str, list[str]]:
    fallback = extract_text_fallback_profile(raw_text)
    merged = {key: list(values) for key, values in profile.items()}

    for field_name, fallback_values in fallback.items():
        current_values = merged.get(field_name, [])
        if field_name == "Name" and fallback_values:
            if (
                not current_values
                or len(current_values[0].split()) < 2
                or looks_like_organization(current_values[0])
                or looks_like_title(current_values[0])
                or current_values[0].isupper()
            ):
                merged[field_name] = fallback_values
            continue

        if field_name == "Skills" and fallback_values:
            filtered_current = refine_skill_values(current_values)
            filtered_fallback = refine_skill_values(fallback_values)
            current_quality = score_skill_list(filtered_current)
            fallback_quality = score_skill_list(filtered_fallback)
            if filtered_current and current_quality >= fallback_quality and len(filtered_current) >= max(3, len(filtered_fallback) // 2):
                merged[field_name] = filtered_current
            elif filtered_fallback:
                merged[field_name] = filtered_fallback
            continue

        if field_name == "Email Address" and fallback_values:
            if (
                not current_values
                or not looks_like_valid_email(current_values[0])
                or current_values[0].count("@") != 1
            ):
                merged[field_name] = fallback_values
            continue

        if field_name == "Phone" and fallback_values:
            if not current_values or not looks_like_valid_phone(current_values[0]):
                merged[field_name] = fallback_values
            continue

        if field_name == "Designation" and fallback_values:
            if (
                not current_values
                or looks_like_organization(current_values[0])
                or len(current_values[0].split()) > 8
                or looks_like_degree(current_values[0])
            ):
                merged[field_name] = fallback_values
            continue

        if field_name == "Companies worked at" and fallback_values:
            if (
                not current_values
                or looks_like_action_sentence(current_values[0])
                or len(current_values[0].split()) > 8
                or looks_like_degree(current_values[0])
            ):
                merged[field_name] = fallback_values
            continue

        if field_name == "Location" and fallback_values:
            current_location = current_values[0] if current_values else ""
            if (
                not current_values
                or any(char.isdigit() for char in current_location)
                or looks_like_name(current_location)
                or not looks_like_location(current_location)
            ):
                merged[field_name] = fallback_values
            continue

        if field_name == "Graduation Year" and fallback_values:
            if not current_values:
                merged[field_name] = fallback_values
                continue

            current_year_match = re.search(r"\b(19|20)\d{2}\b", current_values[0])
            fallback_year_match = re.search(r"\b(19|20)\d{2}\b", fallback_values[0])
            if fallback_year_match and (
                not current_year_match or int(fallback_year_match.group(0)) >= int(current_year_match.group(0))
            ):
                merged[field_name] = fallback_values
            continue

        if field_name == "Languages" and fallback_values:
            if not current_values or len(fallback_values) > len(current_values):
                merged[field_name] = fallback_values
            continue

        if not current_values:
            merged[field_name] = fallback_values

    return merged


def calculate_role_match(profile: dict[str, list[str]], model_confidence: float = 0.70) -> int:
    tracked_fields = [
        "Name",
        "Email Address",
        "Phone",
        "Location",
        "Degree",
        "College Name",
        "Graduation Year",
        "Designation",
        "Companies worked at",
        "Skills",
    ]
    detected_fields = sum(1 for field_name in tracked_fields if profile.get(field_name))
    completeness_ratio = detected_fields / len(tracked_fields) if tracked_fields else 0.0

    skills_count = len(profile.get("Skills", []))
    experience_strength = 0
    if profile.get("Designation"):
        experience_strength += 1
    if profile.get("Companies worked at"):
        experience_strength += 1
    if profile.get("Projects"):
        experience_strength += 1
    if profile.get("Years of Experience"):
        experience_strength += 1

    confidence_ratio = max(0.45, min(model_confidence, 0.99))
    richness_ratio = min(skills_count, 8) / 8 if skills_count else 0.0
    experience_ratio = experience_strength / 4

    score = 0
    score += completeness_ratio * 52
    score += confidence_ratio * 30
    score += richness_ratio * 8
    score += experience_ratio * 8

    if completeness_ratio >= 0.8:
        score += 4
    if confidence_ratio >= 0.8:
        score += 3

    return max(35, min(round(score), 97))


def predict_entities_with_quality(
    resume_text: str,
    model,
    tokenizer,
    device,
    max_length: int = 256,
    section_aware: bool = False,
) -> tuple[dict[str, list[str]], float]:
    """Predict entities and estimate overall model confidence for this resume."""

    id_to_label = get_id_to_label(model)
    extracted: dict[str, list[str]] = {}
    confidence_values: list[float] = []

    segments = (
        split_resume_into_sections(resume_text)
        if section_aware
        else [TextSegment(text=resume_text, start_offset=0, section_name="full_resume")]
    )

    model.eval()

    for segment in segments:
        encoded_chunks = tokenizer(
            segment.text,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_overflowing_tokens=True,
            stride=64,
            return_tensors="pt",
        )

        chunk_count = encoded_chunks["input_ids"].shape[0]
        for chunk_index in range(chunk_count):
            input_ids = encoded_chunks["input_ids"][chunk_index : chunk_index + 1].to(device)
            attention_mask = encoded_chunks["attention_mask"][chunk_index : chunk_index + 1].to(device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probabilities = torch.softmax(outputs.logits, dim=-1)
                predictions = torch.argmax(outputs.logits, dim=-1)[0].cpu().tolist()
                prediction_scores = torch.max(probabilities, dim=-1).values[0].cpu().tolist()

            tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())
            chunk_entities = decode_entities_from_labels(tokens, predictions, id_to_label)

            for token, label_id, score in zip(tokens, predictions, prediction_scores):
                label = id_to_label.get(int(label_id), "O")
                if label != "O" and label_id != -100 and token not in {"[CLS]", "[SEP]", "[PAD]"}:
                    confidence_values.append(float(score))

            for label, values in chunk_entities.items():
                for value in values:
                    clean_value = " ".join(value.split())
                    if clean_value and clean_value not in extracted.setdefault(label, []):
                        extracted[label].append(clean_value)

    average_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.70
    return extracted, average_confidence


def build_ui_profile(profile: dict[str, list[str]], model_confidence: float = 0.70) -> dict[str, Any]:
    languages = profile.get("Languages", [])
    skills = profile.get("Skills", [])
    experience_highlights = profile.get("Experience Highlights", [])
    summary_parts = []
    name = flatten_first(profile, "Name", "Candidate")
    title = flatten_first(profile, "Designation", "Resume Candidate")
    company = flatten_first(profile, "Companies worked at")
    degree = normalize_degree_display(flatten_first(profile, "Degree"))
    major = derive_major_from_degree(degree)
    university = normalize_university_display(flatten_first(profile, "College Name"))
    graduation = flatten_first(profile, "Graduation Year")
    location = flatten_first(profile, "Location")
    about = flatten_first(profile, "About")

    skills = remove_context_tokens_from_skills(skills, name, title, company, location)
    if location in {name, title} or any(token in location.lower() for token in ["lorem", "ipsum", "finallygot", "goddamn"]):
        location = "Not detected"
    if degree == "Not detected" and university == "Not detected":
        graduation = "Not detected"
    if company != "Not detected" and university != "Not detected" and company.lower() == university.lower():
        company = "Not detected"
    if title == "Resume Candidate" and company != "Not detected":
        title = company if looks_like_title(company) else title

    if title != "Resume Candidate":
        summary_parts.append(f"{name} is presented as a {title}.")
    if about != "Not detected":
        summary_parts.append(f"Profile summary: {ensure_sentence(about)}")
    if company != "Not detected":
        summary_parts.append(f"The latest detected company is {company}.")
    if degree != "Not detected" or university != "Not detected":
        education_phrase = degree if degree != "Not detected" else "a detected degree"
        if university != "Not detected":
            education_phrase += f" from {university}"
        if graduation != "Not detected":
            education_phrase += f" ({graduation})"
        summary_parts.append(f"Education points to {education_phrase}.")
    if skills:
        preview = ", ".join(skills[:4])
        summary_parts.append(f"Key extracted skills include {preview}.")
    if experience_highlights:
        summary_parts.append(f"Experience evidence includes: {ensure_sentence(experience_highlights[0])}")
    if languages:
        summary_parts.append(f"Detected languages include {', '.join(languages[:3])}.")
    if location != "Not detected":
        summary_parts.append(f"The resume appears tied to {location}.")

    return {
        "name": name,
        "title": title,
        "roleMatch": calculate_role_match(profile, model_confidence=model_confidence),
        "location": location,
        "email": flatten_first(profile, "Email Address"),
        "phone": flatten_first(profile, "Phone"),
        "skills": skills,
        "education": degree,
        "major": major,
        "university": university,
        "graduation": graduation,
        "company": company,
        "experienceLength": flatten_first(profile, "Years of Experience", "Experience detected"),
        "experienceHighlights": experience_highlights,
        "languages": languages,
        "projects": profile.get("Projects", [])[:3],
        "about": about,
        "summary": " ".join(summary_parts) if summary_parts else "The parser extracted structured resume details from the current document.",
    }


def build_stats(profile: dict[str, list[str]]) -> list[dict[str, Any]]:
    return [
        {"label": "Skill", "value": len(profile.get("Skills", [])), "color": "bg-brand-500"},
        {
            "label": "Education",
            "value": len(profile.get("Degree", []))
            + len(profile.get("College Name", []))
            + len(profile.get("Graduation Year", [])),
            "color": "bg-emerald-500",
        },
        {
            "label": "Experience",
            "value": len(profile.get("Designation", []))
            + len(profile.get("Companies worked at", []))
            + len(profile.get("Years of Experience", [])),
            "color": "bg-orange-400",
        },
        {
            "label": "Contact",
            "value": len(profile.get("Name", []))
            + len(profile.get("Email Address", []))
            + len(profile.get("Phone", []))
            + len(profile.get("Location", [])),
            "color": "bg-violet-500",
        },
    ]


def build_metrics_cards(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    baseline = metrics.get("baseline", {})
    section_aware = metrics.get("section_aware", {})
    baseline_accuracy = baseline.get("accuracy", 0.0)
    section_accuracy = section_aware.get("accuracy", 0.0)
    best_accuracy = max(baseline_accuracy, section_accuracy)
    best_accuracy_key = "section_aware" if section_accuracy >= baseline_accuracy else "baseline"

    return [
        {
            "title": "Baseline Model",
            "value": f"{baseline_accuracy:.2f}",
            "note": "Accuracy",
            "color": "text-brand-500",
        },
        {
            "title": "Section-Aware Model",
            "value": f"{section_accuracy:.2f}",
            "note": "Accuracy",
            "color": "text-emerald-500",
        },
        {
            "title": "Best Accuracy",
            "value": f"{best_accuracy:.2f}",
            "note": "Section-Aware" if best_accuracy_key == "section_aware" else "Baseline",
            "color": "text-brand-500",
        },
        {
            "title": "Best Model",
            "value": "Section-Aware Model" if best_accuracy_key == "section_aware" else "Baseline Model",
            "note": "Highest test accuracy",
            "color": "text-emerald-500",
            "smaller": True,
        },
    ]


def build_resume_snapshot(profile: dict[str, list[str]], parser_key: str) -> list[dict[str, Any]]:
    tracked_fields = [
        "Name",
        "Email Address",
        "Phone",
        "Location",
        "Degree",
        "College Name",
        "Graduation Year",
        "Designation",
        "Companies worked at",
        "Skills",
    ]
    detected_fields = sum(1 for field_name in tracked_fields if profile.get(field_name))
    coverage = round((detected_fields / len(tracked_fields)) * 100) if tracked_fields else 0
    role_match = calculate_role_match(profile)
    skills_count = len(profile.get("Skills", []))

    return [
        {
            "title": "Extraction Coverage",
            "value": f"{coverage}%",
            "note": f"{detected_fields}/{len(tracked_fields)} key fields detected",
            "color": "text-brand-500",
        },
        {
            "title": "Role Match",
            "value": f"{role_match}%",
            "note": "Calculated from current extraction",
            "color": "text-emerald-500",
        },
        {
            "title": "Skills Extracted",
            "value": str(skills_count),
            "note": "Detected skill tags",
            "color": "text-brand-500",
        },
        {
            "title": "Parser Used",
            "value": "Section-Aware Model" if parser_key == "section_aware" else "Baseline Model",
            "note": "Active parser for this resume",
            "color": "text-emerald-500",
            "smaller": True,
        },
    ]


def build_resume_confidence_cards(
    baseline_confidence: float,
    section_confidence: float,
) -> list[dict[str, Any]]:
    best_key = "section_aware" if section_confidence >= baseline_confidence else "baseline"
    best_confidence = max(baseline_confidence, section_confidence)

    return [
        {
            "title": "Baseline Model",
            "value": f"{baseline_confidence:.2f}",
            "note": "Confidence",
            "color": "text-brand-500",
        },
        {
            "title": "Section-Aware Model",
            "value": f"{section_confidence:.2f}",
            "note": "Confidence",
            "color": "text-emerald-500",
        },
        {
            "title": "Best Confidence",
            "value": f"{best_confidence:.2f}",
            "note": "Section-Aware" if best_key == "section_aware" else "Baseline",
            "color": "text-brand-500",
        },
        {
            "title": "Best Model",
            "value": "Section-Aware Model" if best_key == "section_aware" else "Baseline Model",
            "note": "Higher confidence on this resume",
            "color": "text-emerald-500",
            "smaller": True,
        },
    ]


def normalize_question(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "question": item.get("question", "").strip(),
        "type": item.get("type", "General"),
        "basedOn": item.get("based_on") or item.get("basedOn") or "Candidate profile",
        "difficulty": item.get("difficulty", "Medium"),
    }


def normalize_feedback(raw_feedback: dict[str, Any]) -> dict[str, Any]:
    score_value = raw_feedback.get("score_out_of_5", raw_feedback.get("score", 0))
    try:
        score = int(float(score_value))
    except (TypeError, ValueError):
        score = 0

    return {
        "score": max(0, min(score, 5)),
        "strengths": raw_feedback.get("strength", raw_feedback.get("strengths", "")),
        "improvements": raw_feedback.get("improvement", raw_feedback.get("improvements", "")),
        "suggested": raw_feedback.get("suggested_better_answer", raw_feedback.get("suggested", "")),
    }


def prefers_offline(selected_model: str) -> bool:
    ai_available = bool(os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY"))
    return selected_model.strip().lower() == "bert only" or not ai_available


def generate_questions_safe(
    profile: dict[str, list[str]],
    language: str,
    selected_model: str,
    question_count: int,
) -> dict[str, Any]:
    if prefers_offline(selected_model):
        questions = generate_template_questions(profile, language=language, number_of_questions=question_count)
        return {"questions": [normalize_question(item) for item in questions], "source": "offline"}

    try:
        questions = generate_questions(profile, language=language, number_of_questions=question_count)
        return {"questions": [normalize_question(item) for item in questions], "source": "ai"}
    except Exception as error:
        questions = generate_template_questions(profile, language=language, number_of_questions=question_count)
        return {
            "questions": [normalize_question(item) for item in questions],
            "source": "offline_fallback",
            "warning": str(error),
        }


def evaluate_answer_safe(
    question: dict[str, Any],
    answer: str,
    profile: dict[str, list[str]],
    language: str,
    selected_model: str,
) -> dict[str, Any]:
    if prefers_offline(selected_model):
        raw_feedback = evaluate_template_answer(answer, language=language)
        return {"feedback": normalize_feedback(raw_feedback), "source": "offline"}

    api_question = {
        "question": question.get("question", ""),
        "type": question.get("type", "General"),
        "based_on": question.get("basedOn") or question.get("based_on") or "Candidate profile",
    }

    try:
        raw_feedback = evaluate_answer(api_question, answer, profile, language=language)
        return {"feedback": normalize_feedback(raw_feedback), "source": "ai"}
    except Exception as error:
        raw_feedback = evaluate_template_answer(answer, language=language)
        return {
            "feedback": normalize_feedback(raw_feedback),
            "source": "offline_fallback",
            "warning": str(error),
        }


def create_summary_safe(results: list[dict[str, Any]], language: str, selected_model: str) -> dict[str, Any]:
    if prefers_offline(selected_model):
        return {"summary": create_template_summary(results, language=language), "source": "offline"}

    try:
        return {"summary": create_final_summary(results, language=language), "source": "ai"}
    except Exception as error:
        return {
            "summary": create_template_summary(results, language=language),
            "source": "offline_fallback",
            "warning": str(error),
        }


@app.get("/api/bootstrap")
def bootstrap() -> dict[str, Any]:
    return {
        "status": "ok",
        "bestParserModel": BEST_PARSER_KEY,
        "metrics": build_metrics_cards(METRICS),
        "rawMetrics": METRICS,
        "device": str(DEVICE),
        "aiAvailable": bool(os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")),
    }


@app.post("/api/process-resume")
def process_resume(
    file: UploadFile = File(...),
    parser_model: str = BEST_PARSER_KEY,
) -> dict[str, Any]:
    saved_path: Path | None = None
    try:
        saved_path = save_upload(file)
        raw_text = extract_text_from_document(saved_path)
        if not raw_text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from this document.")

        bundle = get_parser_bundle(parser_model)
        baseline_bundle = get_parser_bundle("baseline")
        section_bundle = get_parser_bundle("section_aware")

        baseline_entities, baseline_confidence = predict_entities_with_quality(
            resume_text=raw_text,
            model=baseline_bundle["model"],
            tokenizer=baseline_bundle["tokenizer"],
            device=DEVICE,
            max_length=MAX_LENGTH,
            section_aware=baseline_bundle["section_aware"],
        )
        section_entities, section_confidence = predict_entities_with_quality(
            resume_text=raw_text,
            model=section_bundle["model"],
            tokenizer=section_bundle["tokenizer"],
            device=DEVICE,
            max_length=MAX_LENGTH,
            section_aware=section_bundle["section_aware"],
        )

        if bundle["key"] == "baseline":
            entities = baseline_entities
            model_confidence = baseline_confidence
        else:
            entities = section_entities
            model_confidence = section_confidence

        profile = merge_profile_with_fallback(build_profile_from_entities(entities), raw_text)
        ui_profile = build_ui_profile(profile, model_confidence=model_confidence)

        return {
            "processed": True,
            "extractedText": raw_text,
            "profile": ui_profile,
            "profileRaw": profile,
            "stats": build_stats(profile),
            "metrics": build_resume_confidence_cards(baseline_confidence, section_confidence),
            "parserModel": bundle["key"],
            "parserLabel": "Section-Aware BERT" if bundle["key"] == "section_aware" else "Baseline BERT",
            "fileName": file.filename,
        }
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    finally:
        if saved_path and saved_path.exists():
            try:
                saved_path.unlink()
            except OSError:
                pass


@app.post("/api/start-interview")
def start_interview(request: InterviewStartRequest) -> dict[str, Any]:
    result = generate_questions_safe(
        profile=request.profile,
        language=request.language,
        selected_model=request.selected_model,
        question_count=request.question_count,
    )
    return result


@app.post("/api/evaluate-answer")
def evaluate_interview_answer(request: EvaluateAnswerRequest) -> dict[str, Any]:
    return evaluate_answer_safe(
        question=request.question,
        answer=request.answer,
        profile=request.profile,
        language=request.language,
        selected_model=request.selected_model,
    )


@app.post("/api/final-summary")
def final_summary(request: FinalSummaryRequest) -> dict[str, Any]:
    return create_summary_safe(
        results=request.results,
        language=request.language,
        selected_model=request.selected_model,
    )


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# Serve the React UI after API routes.
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="react-ui")
