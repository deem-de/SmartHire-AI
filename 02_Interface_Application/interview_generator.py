"""Generate interview questions from extracted resume data.

Part 2 stays separate from the BERT parser. It receives a structured candidate
profile and uses a GPT model only to write clear interview questions.
"""

from __future__ import annotations

import json
import os

from openai import OpenAI, OpenAIError

from interview_gemini import generate_with_gemini


DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def format_candidate_profile(profile: dict) -> str:
    """Convert extracted resume fields into readable text for the prompt."""

    lines = []
    for field_name, values in profile.items():
        if not values:
            continue

        if isinstance(values, list):
            clean_values = ", ".join(str(value) for value in values[:8])
        else:
            clean_values = str(values)

        lines.append(f"{field_name}: {clean_values}")

    return "\n".join(lines)


def generate_questions(
    profile: dict,
    language: str = "English",
    number_of_questions: int = 5,
    model: str = DEFAULT_MODEL,
) -> list[dict]:
    """Generate interview questions from resume data.

    The model is instructed to use only the provided profile, which reduces
    hallucination and makes the questions evidence-based.
    """

    profile_text = format_candidate_profile(profile)

    instructions = (
        "You are an AI interview question generator for a university resume "
        "parsing project. Generate practical interview questions using only "
        "the candidate profile provided. Do not invent skills or experience. "
        "Return valid JSON only."
    )

    prompt = f"""
Candidate profile:
{profile_text}

Language: {language}
Number of questions: {number_of_questions}

Create interview questions with these types:
- Technical
- Project-Based
- Behavioral
- Experience

Return a JSON array. Each item must have:
- question
- type
- based_on
- difficulty
"""

    if os.getenv("GEMINI_API_KEY"):
        text = generate_with_gemini(prompt=prompt, instructions=instructions)
        return _parse_json_list(text)

    client = OpenAI()
    try:
        response = client.responses.create(
            model=model,
            instructions=instructions,
            input=prompt,
        )
    except OpenAIError as error:
        raise RuntimeError(
            "OpenAI API call failed while generating questions. "
            "Check your OPENAI_API_KEY, internet connection, account credits, "
            f"and model name. Current model: {model}. Original error: {error}"
        ) from error

    return _parse_json_list(response.output_text)


def _parse_json_list(text: str) -> list[dict]:
    """Parse GPT JSON output with a simple safe fallback."""

    cleaned_text = _remove_json_fence(text)

    try:
        data = json.loads(cleaned_text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    return [
        {
            "question": text.strip(),
            "type": "General",
            "based_on": "Candidate profile",
            "difficulty": "Medium",
        }
    ]


def _remove_json_fence(text: str) -> str:
    """Remove markdown JSON fences if the model includes them."""

    text = text.strip()
    if text.startswith("```json"):
        text = text.removeprefix("```json").removesuffix("```")
    elif text.startswith("```"):
        text = text.removeprefix("```").removesuffix("```")
    return text.strip()
