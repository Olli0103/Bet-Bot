#!/usr/bin/env python3
"""Bootstrap free historical datasets quickly into data/raw/*

Sources:
- football-data.co.uk (CSV) — Soccer leagues with full odds
- tennis-data.co.uk (XLSX/CSV) — ATP, WTA with Pinnacle/bet365 odds
- aussportsbetting.com — NBA, NFL, NHL with moneyline, spreads, totals odds

Usage:
    python scripts/bootstrap_history.py
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
    """Download EPL + Bundesliga latest seasons from football-data.co.uk."""
    urls = {
        "football/epl_2324.csv": "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
        "football/epl_2425.csv": "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
        "football/bundesliga_2324.csv": "https://www.football-data.co.uk/mmz4281/2324/D1.csv",
        "football/bundesliga_2425.csv": "https://www.football-data.co.uk/mmz4281/2425/D1.csv",
        "football/laliga_2324.csv": "https://www.football-data.co.uk/mmz4281/2324/SP1.csv",
        "football/seriea_2324.csv": "https://www.football-data.co.uk/mmz4281/2324/I1.csv",
    }
    for rel, url in urls.items():
        try:
            dl(url, RAW / rel)
        except Exception as e:
            print("skip", rel, type(e).__name__)


def tennis_seed_readme():
    """Write instructions for fetching tennis datasets."""
    p = RAW / "tennis/README.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        "=== Tennis Historical Data ===\n\n"
        "Option 1: tennis-data.co.uk (XLSX, includes Pinnacle/bet365 odds)\n"
        "  Download from: https://www.tennis-data.co.uk/data.php\n"
        "  Files: ATP .xlsx, WTA .xlsx\n\n"
        "Option 2: Jeff Sackmann (CSV, scores only, no odds)\n"
        "  https://github.com/JeffSackmann/tennis_atp\n"
        "  https://github.com/JeffSackmann/tennis_wta\n\n"
        "Place files in data/imports/tennis/\n"
        "The importer auto-detects ATP/WTA/Challenger from filenames.\n",
        encoding="utf-8",
    )
    print("wrote", p)


def us_sports_seed_readme():
    """Write instructions for fetching US sports datasets."""
    p = RAW / "us_sports/README.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        "=== US Sports Historical Data (aussportsbetting.com format) ===\n\n"
        "Download from: https://www.aussportsbetting.com/data/\n\n"
        "NBA:\n"
        "  File: nba.csv (or nba_20XX_XX.csv)\n"
        "  Columns: Date, Home Team, Away Team, Home Score, Away Score,\n"
        "           Home Odds, Away Odds, Home Line, Home Line Odds,\n"
        "           Away Line Odds, Total Score Line, Total Score Over Odds,\n"
        "           Total Score Under Odds\n"
        "  Place in: data/imports/nba/\n\n"
        "NFL:\n"
        "  File: nfl.csv\n"
        "  Same column format as NBA\n"
        "  Place in: data/imports/nfl/\n\n"
        "NHL:\n"
        "  File: nhl.csv\n"
        "  Same column format as NBA\n"
        "  Place in: data/imports/nhl/\n\n"
        "Then run:\n"
        "  python scripts/import_historical_results.py \\\n"
        "    --imports-dir data/imports \\\n"
        "    --nba-dir data/imports/nba \\\n"
        "    --nfl-dir data/imports/nfl \\\n"
        "    --nhl-dir data/imports/nhl\n",
        encoding="utf-8",
    )
    print("wrote", p)


def main():
    football_seed()
    tennis_seed_readme()
    us_sports_seed_readme()
    print("bootstrap_history done")


if __name__ == "__main__":
    main()
