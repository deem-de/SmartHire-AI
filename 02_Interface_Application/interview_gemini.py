"""Gemini API helper for the interview assistant."""

from __future__ import annotations

import os


DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")


def generate_with_gemini(
    prompt: str,
    instructions: str = "",
    model_name: str = DEFAULT_GEMINI_MODEL,
) -> str:
    """Generate text using Gemini.

    This uses the google-generativeai package because it is already installed
    in the project environment.
    """

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing.")

    try:
        import google.generativeai as genai
    except ImportError as error:
        raise RuntimeError(
            "google-generativeai is not installed. Run: "
            "python -m pip install google-generativeai"
        ) from error

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=instructions or None,
        )
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as error:
        raise RuntimeError(
            "Gemini API call failed. Check GEMINI_API_KEY, quota, internet, "
            f"and model name. Current model: {model_name}. Original error: {error}"
        ) from error
