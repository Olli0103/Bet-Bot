"""Ollama LLM client for sentiment analysis and intent classification.

Optimized for Gemma 3 4B: concise prompts, zero temperature, strict JSON output.
"""
import json
from typing import Any, Dict

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.core.settings import settings
from src.models.sentiment import SentimentResult


class OllamaSentimentClient:
    def __init__(self) -> None:
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.model = settings.ollama_model

    @retry(
        retry=retry_if_exception_type(httpx.HTTPError),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def analyze(self, text: str, context: str = "") -> SentimentResult:
        """Run sentiment analysis. Returns structured SentimentResult."""
        prompt = (
            "Du bist Sport-Sentiment-Analyst. "
            "Bewerte den Text: positive, neutral oder negative. "
            'Antworte NUR als JSON: {"label":"...","score":0.0-1.0,"confidence":0.0-1.0,"rationale":"1 Satz"}.\n\n'
            f"Kontext: {context}\n\nText:\n{text}"
        )

        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0},
        }

        with httpx.Client(timeout=90) as client:
            r = client.post(f"{self.base_url}/api/generate", json=payload)
        r.raise_for_status()
        data = r.json()

        raw = (data.get("response") or "{}").strip()
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = {"label": "neutral", "score": 0.5, "confidence": 0.3, "rationale": raw[:240]}

        return SentimentResult(**parsed)

    @retry(
        retry=retry_if_exception_type(httpx.HTTPError),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def generate_raw(self, prompt: str) -> str:
        """Low-level generation with zero temperature. Returns raw text."""
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0},
        }

        with httpx.Client(timeout=90) as client:
            r = client.post(f"{self.base_url}/api/generate", json=payload)
        r.raise_for_status()
        return (r.json().get("response") or "").strip()

    @retry(
        retry=retry_if_exception_type(httpx.HTTPError),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def generate_json(self, prompt: str) -> dict:
        """Generate structured JSON output with zero temperature."""
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0},
        }

        with httpx.Client(timeout=90) as client:
            r = client.post(f"{self.base_url}/api/generate", json=payload)
        r.raise_for_status()
        raw = (r.json().get("response") or "{}").strip()
        try:
            return json.loads(raw)
        except Exception:
            return {}
