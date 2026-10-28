from __future__ import annotations

import json
import os
import re
from typing import Any

import requests


def article_title_to_keywords(title: str, top_n: int = 5) -> list[str]:
    """Extract top PubMed-search keywords/phrases from an article title using Gemini."""
    if not title or not title.strip():
        return []

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing GEMINI_API_KEY. Add it to your .env or environment variables."
        )

    # Default to a gemini model; override via GEMINI_KEYWORD_MODEL if needed.
    model = os.getenv("GEMINI_KEYWORD_MODEL", "gemini-3.8-flash")

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/"
        f"models/{model}:generateContent"
    )

    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": api_key,
    }
        # Request ONLY a JSON array of strings.
    user_prompt = (
        "You are an expert biomedical search assistant.\n"
        "Given the following article title, generate up to {top_n} concise PubMed search keywords/phrases.\n"
        "Prefer key concepts and common synonyms/terms. These should be suitable for building a PubMed query.\n"
        "Return ONLY a valid JSON array of strings (no object wrapper, no extra text).\n\n"
        "TITLE:\n"
        "{title}"
    ).format(top_n=top_n, title=title.strip()[:2000])

    payload = {
        "contents": [{
            "parts": [{"text": user_prompt}]
        }],
        "systemInstruction": {
            "parts": [{"text": "Return only the requested JSON."}]
        },
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 1024,
            "responseMimeType": "application/json",
        }
    }

    resp = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=60,
    )

    if resp.status_code >= 400:
        try:
            err_json = resp.json()
            error_text = json.dumps(err_json)
        except Exception:
            error_text = resp.text

        raise RuntimeError(
            f"Gemini API error ({resp.status_code}) using model '{model}': {error_text}"
        )

    data = resp.json()
    try:
        model_text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        raise RuntimeError("Unexpected response format from Gemini API: " + json.dumps(data))
        
    return _parse_keywords_from_model_text(model_text, top_n=top_n)


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    """Dedupe strings case-insensitively while preserving order."""
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        key = v.strip().casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(v.strip())
    return out


def _parse_keywords_from_model_text(text: str, top_n: int) -> list[str]:
    """
    Best-effort extraction of keyword strings from model output.
    Prefer JSON, but fall back to heuristic parsing if needed.
    """
    if not text:
        return []

    cleaned = text.strip()
    # Remove possible fenced code blocks (```json ... ```).
    cleaned = cleaned.replace("```json", "```").replace("```", "").strip()

    # Try JSON first.
    try:
        obj: Any = json.loads(cleaned)
        if isinstance(obj, list):
            keywords = [str(x) for x in obj]
            return _dedupe_preserve_order([k for k in keywords if k.strip()])[:top_n]
        if isinstance(obj, dict):
            maybe = obj.get("keywords") or obj.get("keyphrases")
            if isinstance(maybe, list):
                keywords = [str(x) for x in maybe]
                return _dedupe_preserve_order([k for k in keywords if k.strip()])[:top_n]
    except Exception:
        pass

    # Heuristic fallback: pick up lines like "1. xxx", bullets, or quoted phrases.
    candidates: list[str] = []
    for line in cleaned.splitlines():
        line = line.strip()
        if not line:
            continue
        # Remove common numbering/bullets.
        line = re.sub(r"^[-*]\s+", "", line)
        line = re.sub(r"^\d+\.\s+", "", line)
        line = re.sub(r"^\(\d+\)\s+", "", line)
        # Remove surrounding quotes/backticks.
        line = line.strip("`\"' ")
        if line:
            candidates.append(line)

    return _dedupe_preserve_order(candidates)[:top_n]


def select_keywords(keywords: list[str]) -> list[str]:
    """Prompt the user to choose one or more extracted keywords by index."""
    if not keywords:
        return []

    print("Extracted Keywords:")
    for index, keyword in enumerate(keywords, start=1):
        print(f"{index}. {keyword}")

    while True:
        raw_selection = input("Select keywords by index (comma-separated): ").strip()
        if not raw_selection:
            print("Please select at least one keyword.")
            continue

        try:
            selected_indices = [
                int(part.strip())
                for part in raw_selection.split(",")
                if part.strip()
            ]
        except ValueError:
            print("Invalid input. Enter comma-separated numbers like: 1,2,3")
            continue

        if not selected_indices:
            print("Please select at least one keyword.")
            continue

        invalid_indices = [
            index
            for index in selected_indices
            if index < 1 or index > len(keywords)
        ]
        if invalid_indices:
            print(
                "Invalid keyword index: "
                + ", ".join(str(index) for index in invalid_indices)
            )
            continue

        selected_keywords: list[str] = []
        for index in selected_indices:
            keyword = keywords[index - 1]
            if keyword not in selected_keywords:
                selected_keywords.append(keyword)

        return selected_keywords
