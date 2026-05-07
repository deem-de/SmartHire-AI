"""Evaluate candidate answers during the terminal interview.

This file uses a GPT model to give simple feedback with a score. The scoring
rubric is fixed by our code so the feedback is easier to explain in a demo.
"""

from __future__ import annotations

import json
import os

from openai import OpenAI, OpenAIError

from interview_gemini import generate_with_gemini


DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def evaluate_answer(
    question: dict,
    answer: str,
    profile: dict,
    language: str = "English",
    model: str = DEFAULT_MODEL,
) -> dict:
    """Evaluate one user answer and return score plus feedback."""

    instructions = (
        "You are an AI interview evaluator. Be fair, concise, and supportive. "
        "Evaluate the answer using the rubric. Do not be harsh. Return valid "
        "JSON only."
    )

    prompt = f"""
Candidate profile:
{json.dumps(profile, ensure_ascii=False)}

Interview language: {language}

Question:
{question.get("question", "")}

Question type:
{question.get("type", "General")}

Question was based on:
{question.get("based_on", "Candidate profile")}

Candidate answer:
{answer}

Evaluate using this 5-point rubric:
- Technical correctness
- Clarity
- Completeness
- Use of example

Return a JSON object with:
- score_out_of_5
- strength
- improvement
- suggested_better_answer
"""

    if os.getenv("GEMINI_API_KEY"):
        text = generate_with_gemini(prompt=prompt, instructions=instructions)
        return _parse_json_object(text)

    client = OpenAI()
    try:
        response = client.responses.create(
            model=model,
            instructions=instructions,
            input=prompt,
        )
    except OpenAIError as error:
        raise RuntimeError(
            "OpenAI API call failed while evaluating the answer. "
            "Check your OPENAI_API_KEY, internet connection, account credits, "
            f"and model name. Current model: {model}. Original error: {error}"
        ) from error

    return _parse_json_object(response.output_text)


def create_final_summary(
    interview_results: list[dict],
    language: str = "English",
    model: str = DEFAULT_MODEL,
) -> str:
    """Create a short final interview summary."""

    prompt = f"""
Interview language: {language}

Interview results:
{json.dumps(interview_results, ensure_ascii=False)}

Create a short final interview report with:
- Overall performance
- Main strengths
- Areas to improve
- Recommendation
"""

    if os.getenv("GEMINI_API_KEY"):
        return generate_with_gemini(prompt=prompt).strip()

    client = OpenAI()
    try:
        response = client.responses.create(
            model=model,
            input=prompt,
        )
    except OpenAIError as error:
        raise RuntimeError(
            "OpenAI API call failed while creating the final summary. "
            "Check your OPENAI_API_KEY, internet connection, account credits, "
            f"and model name. Current model: {model}. Original error: {error}"
        ) from error
    return response.output_text.strip()


def _parse_json_object(text: str) -> dict:
    """Parse GPT JSON output with a simple safe fallback."""

    cleaned_text = _remove_json_fence(text)

    try:
        data = json.loads(cleaned_text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    return {
        "score_out_of_5": "N/A",
        "strength": "Feedback was generated, but not in JSON format.",
        "improvement": text.strip(),
        "suggested_better_answer": "",
    }


def _remove_json_fence(text: str) -> str:
    """Remove markdown JSON fences if the model includes them."""

    text = text.strip()
    if text.startswith("```json"):
        text = text.removeprefix("```json").removesuffix("```")
    elif text.startswith("```"):
        text = text.removeprefix("```").removesuffix("```")
    return text.strip()
