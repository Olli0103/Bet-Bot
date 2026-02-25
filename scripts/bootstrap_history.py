#!/usr/bin/env python3
"""
Bootstrap free historical datasets quickly into data/raw/*
Sources:
- football-data.co.uk (CSV)
- Jeff Sackmann tennis (matches/rankings links placeholder)
"""

from pathlib import Path
import urllib.request

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data" / "raw"


def dl(url: str, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as r:
        out.write_bytes(r.read())
    print("downloaded", out)


def football_seed():
    # examples: EPL + Bundesliga latest seasons
    urls = {
        "football/epl_2324.csv": "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
        "football/epl_2425.csv": "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
        "football/bundesliga_2324.csv": "https://www.football-data.co.uk/mmz4281/2324/D1.csv",
        "football/bundesliga_2425.csv": "https://www.football-data.co.uk/mmz4281/2425/D1.csv",
    }
    for rel, url in urls.items():
        try:
            dl(url, RAW / rel)
        except Exception as e:
            print("skip", rel, type(e).__name__)


def tennis_seed_readme():
    p = RAW / "tennis/README.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        "Fetch Jeff Sackmann datasets (manual due repo size):\n"
        "https://github.com/JeffSackmann/tennis_atp\n"
        "https://github.com/JeffSackmann/tennis_wta\n"
        "\n"
        "Optional odds history:\n"
        "https://www.tennis-data.co.uk/data.php\n",
        encoding="utf-8",
    )
    print("wrote", p)


def main():
    football_seed()
    tennis_seed_readme()
    print("bootstrap_history done")


if __name__ == "__main__":
    main()
