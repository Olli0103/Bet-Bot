#!/usr/bin/env python3
"""
Jeff Sackmann ATP Stats Ingester

Downloads historical ATP match data and calculates player-level baselines:
- Hold Rate: % of service games held
- Break Rate: % of opponent's service games broken
- Aces per match, Double Faults per match
- First Serve %, Second Serve Won %

Output: atp_player_stats.json (indexed by normalized player name)

Usage:
    python sackmann_ingester.py [--seasons 2020-2025] [--output data/atp_player_stats.json]
"""
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional
import requests
import pandas as pd
import hashlib

SACKMANN_BASE = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master"

# Columns we need from matches CSV
MATCH_COLS = [
    "winner_name", "loser_name",
    "winner_id", "loser_id",
    "surface", "tourney_level",
    "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon", "w_SvGms", "w_bpSaved", "w_bpFaced",
    "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon", "l_SvGms", "l_bpSaved", "l_bpFaced",
]


def download_matches(year: int, force: bool = False) -> Optional[pd.DataFrame]:
    """Download ATP matches CSV for a given year. Returns None on 404 (year not available)."""
    path = Path("data/sackmann") / f"atp_matches_{year}.csv"
    
    if not force and path.exists():
        print(f"  Using cached: {path}")
        return pd.read_csv(path, low_memory=False)
    
    # Try master branch first, then main branch
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    urls = [
        f"https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv",
        f"https://raw.githubusercontent.com/JeffSackmann/tennis_atp/main/atp_matches_{year}.csv",
    ]
    
    for url in urls:
        print(f"  Downloading: {url}")
        try:
            response = requests.get(url, headers=headers, timeout=30)
            if response.status_code == 200:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(response.text)
                return pd.read_csv(path, low_memory=False)
            elif response.status_code == 404:
                continue  # Try next URL
            else:
                print(f"  HTTP {response.status_code}, trying next...")
        except Exception as e:
            print(f"  Error: {e}, trying next...")
    
    print(f"  ⚠️  {year} data not available (all URLs failed)")
    return None


def normalize_name(name: str) -> str:
    """Normalize player name for matching."""
    if pd.isna(name):
        return ""
    # Lowercase, remove dots, extra spaces
    name = str(name).lower().strip()
    name = re.sub(r'[.\-]', ' ', name)
    name = re.sub(r'\s+', ' ', name)
    return name


def calculate_player_stats(df: pd.DataFrame) -> Dict:
    """
    Calculate career/rolling stats per player from match data.
    
    For each player (as winner OR loser), calculate:
    - hold_rate: service games held / total service games
    - break_rate: opp service games broken / opp total service games
    - ace_rate: aces per serve game
    - df_rate: double faults per serve game
    - first_serve_pct: first serves in / total serves
    - second_win_pct: second serve points won / second serve points
    """
    from collections import defaultdict
    
    stats = defaultdict(lambda: {
        "as_server": {"held": 0, "total": 0},      # Player serving
        "as_returner": {"broken": 0, "total": 0},  # Player returning
        "aces": 0, "dfs": 0, "svpt": 0,
        "first_in": 0, "first_won": 0,
        "second_won": 0, "matches": 0,
        "surfaces": defaultdict(int),
        "levels": defaultdict(int),
    })
    
    for _, row in df.iterrows():
        # --- WINNER stats ---
        w_name = normalize_name(row.get("winner_name", ""))
        if not w_name:
            continue
            
        # As server (winner served)
        w_sv_gms = float(row.get("w_SvGms")) if pd.notna(row.get("w_SvGms")) else 0
        w_bp_faced = float(row.get("w_bpFaced")) if pd.notna(row.get("w_bpFaced")) else 0
        if w_sv_gms + w_bp_faced > 0:
            stats[w_name]["as_server"]["total"] += w_sv_gms + w_bp_faced
            stats[w_name]["as_server"]["held"] += w_sv_gms
        
        # As returner (winner broke loser's serve)
        l_sv_gms = float(row.get("l_SvGms")) if pd.notna(row.get("l_SvGms")) else 0
        l_bp_saved = float(row.get("l_bpSaved")) if pd.notna(row.get("l_bpSaved")) else 0
        l_bp_faced = float(row.get("l_bpFaced")) if pd.notna(row.get("l_bpFaced")) else 0
        if l_sv_gms + l_bp_saved > 0:
            stats[w_name]["as_returner"]["total"] += l_sv_gms + l_bp_saved
        if l_bp_faced > 0:
            stats[w_name]["as_returner"]["broken"] += l_bp_faced  # losers bpFaced = we broke them
        
        # Serve stats
        stats[w_name]["aces"] += float(row.get("w_ace", 0) or 0)
        stats[w_name]["dfs"] += float(row.get("w_df", 0) or 0)
        stats[w_name]["svpt"] += float(row.get("w_svpt", 0) or 0)
        stats[w_name]["first_in"] += float(row.get("w_1stIn", 0) or 0)
        stats[w_name]["first_won"] += float(row.get("w_1stWon", 0) or 0)
        stats[w_name]["second_won"] += float(row.get("w_2ndWon", 0) or 0)
        stats[w_name]["matches"] += 1
        
        # Context
        surf = str(row.get("surface", "")).lower()
        level = str(row.get("tourney_level", "")).upper()
        if surf and surf != "nan":
            stats[w_name]["surfaces"][surf] += 1
        if level and level != "NAN":
            stats[w_name]["levels"][level] += 1
        
        # --- LOSER stats ---
        l_name = normalize_name(row.get("loser_name", ""))
        if not l_name:
            continue
            
        # As server (loser served)
        l_sv_gms = float(row.get("l_SvGms")) if pd.notna(row.get("l_SvGms")) else 0
        l_bp_faced = float(row.get("l_bpFaced")) if pd.notna(row.get("l_bpFaced")) else 0
        stats[l_name]["as_server"]["total"] += l_sv_gms + l_bp_faced
        stats[l_name]["as_server"]["held"] += l_sv_gms
        
        # As returner (loser returned, winner held serve)
        w_sv_gms = float(row.get("w_SvGms")) if pd.notna(row.get("w_SvGms")) else 0
        w_bp_saved = float(row.get("w_bpSaved")) if pd.notna(row.get("w_bpSaved")) else 0
        w_bp_faced = float(row.get("w_bpFaced")) if pd.notna(row.get("w_bpFaced")) else 0
        if w_sv_gms + w_bp_saved > 0:
            stats[l_name]["as_returner"]["total"] += w_sv_gms + w_bp_saved
        if w_bp_faced > 0:
            stats[l_name]["as_returner"]["broken"] += w_bp_faced
        
        # Serve stats
        stats[l_name]["aces"] += float(row.get("l_ace", 0) or 0)
        stats[l_name]["dfs"] += float(row.get("l_df", 0) or 0)
        stats[l_name]["svpt"] += float(row.get("l_svpt", 0) or 0)
        stats[l_name]["first_in"] += float(row.get("l_1stIn", 0) or 0)
        stats[l_name]["first_won"] += float(row.get("l_1stWon", 0) or 0)
        stats[l_name]["second_won"] += float(row.get("l_2ndWon", 0) or 0)
        stats[l_name]["matches"] += 1
        
        # Context
        if surf:
            stats[l_name]["surfaces"][surf] += 1
        if level:
            stats[l_name]["levels"][level] += 1
    
    # Aggregate into final format
    player_stats = {}
    for player, data in stats.items():
        if data["matches"] < 5:  # Minimum sample size
            continue
        
        # Hold Rate = held / total as server
        server_total = data["as_server"]["total"]
        hold_rate = data["as_server"]["held"] / server_total if server_total > 0 else 0.0
        
        # Break Rate = broken / total as returner
        return_total = data["as_returner"]["total"]
        break_rate = data["as_returner"]["broken"] / return_total if return_total > 0 else 0.0
        
        # Serve percentages
        svpt = data["svpt"]
        first_in = data["first_in"]
        first_won = data["first_won"]
        second_won = data["second_won"]
        
        first_serve_pct = first_in / svpt if svpt > 0 else 0.0
        first_win_pct = first_won / first_in if first_in > 0 else 0.0
        second_win_pct = second_won / (svpt - first_in) if (svpt - first_in) > 0 else 0.0
        
        # Aces/DF per match
        ace_rate = data["aces"] / data["matches"]
        df_rate = data["dfs"] / data["matches"]
        
        # Dominant surface
        dominant_surface = max(data["surfaces"].items(), key=lambda x: x[1])[0] if data["surfaces"] else "hard"
        
        player_stats[player] = {
            "matches": data["matches"],
            "hold_rate": round(hold_rate, 4),
            "break_rate": round(break_rate, 4),
            "net_rate": round(hold_rate + break_rate - 1.0, 4),  # >0 = aggressive returner
            "first_serve_pct": round(first_serve_pct, 4),
            "first_win_pct": round(first_win_pct, 4),
            "second_win_pct": round(second_win_pct, 4),
            "ace_rate": round(ace_rate, 3),
            "df_rate": round(df_rate, 3),
            "dominant_surface": dominant_surface,
            "levels": dict(data["levels"]),
        }
    
    return player_stats


def build_name_lookup(player_stats: Dict) -> Dict:
    """Build alternate name lookup (for matching against Odds API names)."""
    lookup = {}
    for name, stats in player_stats.items():
        # Add variations: first-last, last-first, last only
        parts = name.split()
        if len(parts) >= 2:
            lookup[name] = name
            lookup[parts[-1]] = name  # Last name
            lookup["".join(parts)] = name  # Without space
    return lookup


def main():
    import datetime
    
    parser = argparse.ArgumentParser(description="Jeff Sackmann ATP Stats Ingester")
    parser.add_argument("--seasons", default="auto", help="Season range, e.g. 2020-2025, or 'auto' for 1991-current year")
    parser.add_argument("--output", default="data/atp_player_stats.json", help="Output JSON path")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    args = parser.parse_args()
    
    # Parse seasons
    current_year = datetime.datetime.now().year
    
    if args.seasons.lower() == "auto":
        # Dynamic: 1991 to current year
        start_year = 1991
        end_year = current_year
        print(f"📅 Auto mode: fetching 1991-{current_year}")
    elif "-" in args.seasons:
        start_year, end_year = map(int, args.seasons.split("-"))
    else:
        start_year = end_year = int(args.seasons)
    
    years = list(range(start_year, end_year + 1))
    
    print(f"📥 Downloading ATP matches for years: {years}")
    
    # Download and combine
    all_dfs = []
    for year in years:
        df = download_matches(year, force=args.force)
        if df is not None and not df.empty:
            all_dfs.append(df)
            print(f"  {year}: {len(df)} matches")
    
    if not all_dfs:
        print("❌ No data downloaded!")
        return
    
    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"📊 Total matches: {len(combined)}")
    
    # Calculate stats
    print("🧮 Calculating player stats...")
    player_stats = calculate_player_stats(combined)
    print(f"👥 Players with 5+ matches: {len(player_stats)}")
    
    # Build lookup and save
    lookup = build_name_lookup(player_stats)
    
    output = {
        "meta": {
            "seasons": years,
            "total_matches": int(len(combined)),
            "players": len(player_stats),
            "generated_at": pd.Timestamp.now().isoformat(),
        },
        "stats": player_stats,
        "lookup": lookup,
    }
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"✅ Saved to: {args.output}")
    
    # Sample output
    sample_players = ["novak djokovic", "rafael nad", "roger feder", "carlos alcaraz"]
    print("\n📋 Sample stats:")
    for p in sample_players:
        if p in player_stats:
            s = player_stats[p]
            print(f"  {p}: hold={s['hold_rate']:.1%} break={s['break_rate']:.1%} net={s['net_rate']:+.1%} matches={s['matches']}")


if __name__ == "__main__":
    main()
