from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests


class LLMError(Exception):
    pass


@dataclass
class LLMResponse:
    provider: str
    model: str
    text: str
    latency_s: float
    raw: Optional[Dict[str, Any]] = None


def env_has_groq() -> bool:
    return bool(os.getenv("GROQ_API_KEY", "").strip())


def env_has_gemini() -> bool:
    return bool(os.getenv("GEMINI_API_KEY", "").strip())


def call_llm(
    provider: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    max_output_tokens: int = 400,
    timeout_s: int = 60,
    retries: int = 2,
) -> LLMResponse:
    provider = (provider or "").strip().lower()
    if provider == "auto":
        if env_has_groq():
            provider = "groq"
        elif env_has_gemini():
            provider = "gemini"
        else:
            raise LLMError("No API key found. Set GROQ_API_KEY or GEMINI_API_KEY in Space secrets.")

    if provider == "groq":
        return _call_groq(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            timeout_s=timeout_s,
            retries=retries,
        )
    if provider == "gemini":
        return _call_gemini(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            timeout_s=timeout_s,
            retries=retries,
        )

    raise LLMError(f"Unknown provider: {provider}")


def _call_groq(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_output_tokens: int,
    timeout_s: int,
    retries: int,
) -> LLMResponse:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise LLMError("Missing GROQ_API_KEY.")

    # Groq OpenAI-compatible endpoint
    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": float(temperature),
        # Groq docs: prefer max_completion_tokens
        "max_completion_tokens": int(max_output_tokens),
    }

    t0 = time.time()
    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
            if resp.status_code >= 400:
                raise LLMError(f"Groq HTTP {resp.status_code}: {resp.text[:800]}")
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            return LLMResponse(
                provider="groq",
                model=model,
                text=text.strip(),
                latency_s=time.time() - t0,
                raw=data,
            )
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(0.7 * (attempt + 1))
            else:
                break
    raise LLMError(f"Groq call failed: {last_err}")


def _call_gemini(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_output_tokens: int,
    timeout_s: int,
    retries: int,
) -> LLMResponse:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise LLMError("Missing GEMINI_API_KEY.")

    # Gemini REST generateContent
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    params = {"key": api_key}

    # To avoid JSON field naming pitfalls across SDKs, we embed system instructions into the single prompt.
    full_prompt = f"{system_prompt}\n\n{user_prompt}"

    payload = {
        "contents": [
            {
                "parts": [{"text": full_prompt}],
            }
        ],
        "generationConfig": {
            "temperature": float(temperature),
            "maxOutputTokens": int(max_output_tokens),
        },
    }

    t0 = time.time()
    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(url, params=params, json=payload, timeout=timeout_s)
            if resp.status_code >= 400:
                raise LLMError(f"Gemini HTTP {resp.status_code}: {resp.text[:800]}")
            data = resp.json()

            # Typical shape: candidates[0].content.parts[0].text
            candidates = data.get("candidates", [])
            if not candidates:
                raise LLMError(f"Gemini returned no candidates: {str(data)[:800]}")

            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if not parts:
                raise LLMError(f"Gemini returned empty parts: {str(data)[:800]}")

            text = parts[0].get("text", "")
            return LLMResponse(
                provider="gemini",
                model=model,
                text=(text or "").strip(),
                latency_s=time.time() - t0,
                raw=data,
            )
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(0.7 * (attempt + 1))
            else:
                break

    raise LLMError(f"Gemini call failed: {last_err}")