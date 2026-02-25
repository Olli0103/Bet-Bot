import json
from typing import Dict, Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.core.settings import settings
from src.models.sentiment import SentimentResult


class OllamaSentimentClient:
    def __init__(self):
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.model = settings.ollama_model

    @retry(
        retry=retry_if_exception_type(httpx.HTTPError),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def analyze(self, text: str, context: str = "") -> SentimentResult:
        prompt = (
            "Du bist ein Sport-News-Sentiment-Analyst. "
            "Bewerte den Text für Team-/Spieler-Performance-Auswirkung. "
            "Antworte nur als JSON mit Keys: label, score, confidence, rationale. "
            "label muss eines von positive|neutral|negative sein. "
            "score und confidence in [0,1].\n\n"
            f"Kontext: {context}\n\nText:\n{text}"
        )

        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
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
