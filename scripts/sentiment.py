#!/usr/bin/env python3
import argparse
from typing import List, Dict

from src.integrations.news_fetcher import NewsFetcher
from src.integrations.ollama_sentiment import OllamaSentimentClient


def build_rows(team: str, articles: List[Dict], nlp: OllamaSentimentClient):
    rows = []
    for a in articles:
        title = a.get("title") or ""
        desc = a.get("description") or ""
        content = a.get("content") or ""
        source = (a.get("source") or {}).get("name", "")
        published = a.get("publishedAt") or ""
        text = "\n".join([title, desc, content]).strip()
        if not text:
            continue

        sentiment = nlp.analyze(text=text, context=f"Team={team}")
        rows.append(
            {
                "team": team,
                "title": title,
                "source": source,
                "published_at": published,
                "label": sentiment.label,
                "score": sentiment.score,
                "confidence": sentiment.confidence,
                "rationale": sentiment.rationale,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--team", required=True)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    news = NewsFetcher()
    nlp = OllamaSentimentClient()

    data = news.search_news(query=args.team)
    articles = (data.get("articles") or [])[: args.limit]
    rows = build_rows(args.team, articles, nlp)

    print(f"sentiment_rows={len(rows)}")
    for r in rows[:5]:
        print(f"[{r['label']}] {r['score']:.2f} {r['team']} | {r['title'][:80]}")


if __name__ == "__main__":
    main()
