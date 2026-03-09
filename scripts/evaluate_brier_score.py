#!/usr/bin/env python3
"""
Tennis ATP Model Evaluation - Brier Score Showdown

Compares:
- Baseline (Bookmaker implied probability)
- Champion (existing model)
- Challenger (new filtered model)
"""
import sys
import os
from pathlib import Path

BOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BOT_DIR))

import json
import numpy as np
import pandas as pd
from dotenv import load_dotenv
load_dotenv(BOT_DIR / ".env")

from src.data.postgres import SessionLocal
from sqlalchemy import text
from src.core.feature_engineering import enrich_tennis_atp
import joblib


def load_filtered_stats():
    """Load player stats (use main file with 395 players, not filtered)"""
    # Use main stats file - it has better coverage
    with open(BOT_DIR / "data/atp_player_stats.json") as f:
        return json.load(f)


def calculate_brier_score(pred_probs, actuals):
    """Brier Score = mean((predicted_prob - actual)^2)"""
    return np.mean((np.array(pred_probs) - np.array(actuals)) ** 2)


def calculate_log_loss(pred_probs, actuals, eps=1e-15):
    """Log Loss = -mean(y*log(p) + (1-y)*log(1-p))"""
    pred_probs = np.clip(pred_probs, eps, 1 - eps)
    return -np.mean(actuals * np.log(pred_probs) + (1 - actuals) * np.log(1 - pred_probs))


def remove_vig(implied_probs):
    """Remove bookmaker vig to get fair probabilities."""
    total = sum(implied_probs)
    if total <= 0:
        return [0.5, 0.5]
    return [p / total for p in implied_probs]


def load_test_data(limit=10000):
    """Load test data from database."""
    session = SessionLocal()
    
    query = f"""
        SELECT 
            pb.event_id,
            pb.selection,
            pb.odds,
            pb.status,
            pb.meta_features,
            pb.created_at
        FROM placed_bets pb
        WHERE pb.sport = 'tennis_atp'
          AND pb.market = 'h2h'
          AND pb.status IN ('won', 'lost')
          AND pb.is_training_data = TRUE
        ORDER BY pb.created_at DESC
        LIMIT {limit}
    """
    
    df = pd.read_sql(query, session.connection())
    session.close()
    
    # Reconstruct matches
    matches = []
    for event_id, group in df.groupby('event_id'):
        if len(group) != 2:
            continue
        
        rows = group.sort_values('odds').to_dict('records')
        
        # Determine winner
        winner = None
        for row in rows:
            if row['status'] == 'won':
                winner = row['selection']
                winner_odds = row['odds']
                loser_odds = [r['odds'] for r in rows if r['status'] == 'lost'][0]
                break
        
        if not winner:
            continue
        
        # Parse player names - format is "Player Name"
        winner_name = winner
        loser_name = [r['selection'] for r in rows if r['status'] == 'lost'][0]
        
        # Home is lower odds (favorite) in our DB
        home_player = rows[0]['selection']  # Lower odds
        away_player = rows[1]['selection']  # Higher odds
        
        # Target: 1 if home won
        home_won = 1 if winner == home_player else 0
        
        matches.append({
            'event_id': event_id,
            'home_player': home_player,
            'away_player': away_player,
            'home_odds': rows[0]['odds'],
            'away_odds': rows[1]['odds'],
            'home_won': home_won,
            'winner': winner,
        })
    
    return pd.DataFrame(matches)


def get_features(home, away, stats_data, home_odds=None, away_odds=None):
    """Get Jeff Sackmann features for a match - uses proper name matching."""
    import sys
    from pathlib import Path
    BOT_DIR = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(BOT_DIR))
    
    from src.core.feature_engineering import _resolve_player_name, _load_atp_player_stats
    
    # Use the proper name resolution from feature_engineering
    home_stats = _resolve_player_name(home, stats_data)
    away_stats = _resolve_player_name(away, stats_data)
    
    if not home_stats or not away_stats:
        return None
    
    home_rank = 500 - min(home_stats.get('matches', 0) * 2, 400)
    away_rank = 500 - min(away_stats.get('matches', 0) * 2, 400)
    
    feats = {
        'tennis_diff_hold_rate': home_stats.get('hold_rate', 0.65) - away_stats.get('hold_rate', 0.65),
        'tennis_diff_break_rate': home_stats.get('break_rate', 0.35) - away_stats.get('break_rate', 0.35),
        'tennis_diff_net_rate': home_stats.get('net_rate', 0) - away_stats.get('net_rate', 0),
        'has_deep_tennis_stats': 1.0,
        'tennis_home_matches': home_stats.get('matches', 0),
        'tennis_away_matches': away_stats.get('matches', 0),
        'tennis_total_matches': home_stats.get('matches', 0) + away_stats.get('matches', 0),
        'home_rank': home_rank,
        'away_rank': away_rank,
        'rank_diff': home_rank - away_rank,
        'rank_log_diff': np.log1p(home_rank) - np.log1p(away_rank),
        'home_rank_pct': min(home_rank / 500, 1.0),
        'away_rank_pct': min(away_rank / 500, 1.0),
        'form_winrate_l5': 0.5,
        'form_games_l5': 0,
        'has_form': 0.0,
        'sharp_vig': 0.025,
    }
    
    # NO odds features - only fundamental (matching training)
    # if home_odds and away_odds:
    #     implied_home = 1 / home_odds
    #     implied_away = 1 / away_odds
    #     total = implied_home + implied_away
    #     fair_home = implied_home / total
    #     feats['home_implied_prob'] = fair_home
    #     feats['clv'] = fair_home - implied_home
    
    return feats


def main():
    print("=" * 70)
    print("🎾 TENNIS ATP MODEL EVALUATION - BRIER SCORE SHOWDOWN")
    print("=" * 70)
    
    # Load test data
    print("\n[1/4] Loading test data from database...")
    test_data = load_test_data(10000)
    print(f"    Loaded {len(test_data)} matches")
    
    # Load player stats
    print("\n[2/4] Loading player stats...")
    stats_data = load_filtered_stats()
    print(f"    {len(stats_data.get('stats', {}))} players")
    
    # Prepare features
    print("\n[3/4] Computing features...")
    valid_matches = []
    for _, row in test_data.iterrows():
        feats = get_features(row['home_player'], row['away_player'], stats_data, row['home_odds'], row['away_odds'])
        if feats:
            valid_matches.append({
                **row.to_dict(),
                **feats
            })
    
    df = pd.DataFrame(valid_matches)
    print(f"    {len(df)} matches with valid features")
    
    # Apply filter (>=5 matches per player, same as training)
    min_matches = 5
    filter_mask = (df['tennis_home_matches'] >= min_matches) & (df['tennis_away_matches'] >= min_matches)
    df = df[filter_mask]
    print(f"    🎯 {len(df)} matches after >= {min_matches} matches filter")
    
    # NO randomization - training uses full mirror, so evaluation should use original data
    
    if len(df) < 100:
        print("    ⚠️ Not enough data!")
        return
    
    # Calculate Baseline (Bookmaker)
    print("\n[4/4] Computing scores...")
    
    # Bookmaker implied probs (remove vig)
    implied_home = [1 / o for o in df['home_odds']]
    implied_away = [1 / o for o in df['away_odds']]
    fair_probs = [remove_vig([h, a])[0] for h, a in zip(implied_home, implied_away)]
    
    baseline_brier = calculate_brier_score(fair_probs, df['home_won'])
    baseline_log_loss = calculate_log_loss(fair_probs, df['home_won'])
    
    # Champion model (new - with Jeff Sackmann features)
    champion_path = BOT_DIR / "models" / "tennis_atp" / "xgb_tennis_atp.joblib"
    champion_brier = None
    champion_log_loss = None
    
    if champion_path.exists():
        try:
            model = joblib.load(champion_path)
            if hasattr(model, 'predict_proba'):
                # Use same features as training (fundamental only, NO odds)
                feat_cols = ['tennis_diff_hold_rate', 'tennis_diff_break_rate', 'tennis_diff_net_rate',
                           'has_deep_tennis_stats', 'tennis_home_matches', 'tennis_away_matches',
                           'tennis_total_matches', 'home_rank', 'away_rank', 'rank_diff',
                           'rank_log_diff', 'home_rank_pct', 'away_rank_pct',
                           'form_winrate_l5', 'form_games_l5', 'has_form', 'sharp_vig']
                
                X = df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
                probs = model.predict_proba(X)[:, 1]
                
                champion_brier = calculate_brier_score(probs, df['home_won'])
                champion_log_loss = calculate_log_loss(probs, df['home_won'])
        except Exception as e:
            print(f"    Champion load error: {e}")
    
    # Challenger model (new filtered - hybrid)
    challenger_path = BOT_DIR / "models" / "tennis_atp" / "xgb_challenger.joblib"
    challenger_brier = None
    challenger_log_loss = None
    
    print(f"Trying challenger from {challenger_path}")
    if challenger_path.exists():
        try:
            model = joblib.load(challenger_path)
            if hasattr(model, 'predict_proba'):
                # Same features as training (fundamental only, NO odds)
                feat_cols = ['tennis_diff_hold_rate', 'tennis_diff_break_rate', 'tennis_diff_net_rate',
                           'has_deep_tennis_stats', 'tennis_home_matches', 'tennis_away_matches',
                           'tennis_total_matches', 'home_rank', 'away_rank', 'rank_diff',
                           'rank_log_diff', 'home_rank_pct', 'away_rank_pct',
                           'form_winrate_l5', 'form_games_l5', 'has_form', 'sharp_vig']
                
                X = df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
                probs = model.predict_proba(X)[:, 1]
                
                challenger_brier = calculate_brier_score(probs, df['home_won'])
                challenger_log_loss = calculate_log_loss(probs, df['home_won'])
        except Exception as e:
            print(f"    Challenger load error: {e}")
    
    # Print results
    print("\n" + "=" * 70)
    print("📊 MODEL SHOWDOWN RESULTS")
    print("=" * 70)
    print(f"{'Model':<20} {'Brier Score':<15} {'Log Loss':<15}")
    print("-" * 50)
    print(f"{'Bookmaker':<20} {baseline_brier:.4f}         {baseline_log_loss:.4f}")
    if champion_brier:
        print(f"{'Champion':<20} {champion_brier:.4f}         {champion_log_loss:.4f}")
    else:
        print(f"{'Champion':<20} N/A            N/A")
    if challenger_brier:
        print(f"{'Challenger':<20} {challenger_brier:.4f}         {challenger_log_loss:.4f}")
    else:
        print(f"{'Challenger':<20} (not trained yet)")
    print("-" * 50)
    
    # Winner
    if champion_brier and challenger_brier:
        if challenger_brier < champion_brier:
            print(f"\n🏆 CHALLENGER WINS! (Brier improvement: {(champion_brier - challenger_brier)*100:.2f}%)")
        else:
            print(f"\n🏆 CHAMPION WINS! (Brier difference: {(challenger_brier - champion_brier)*100:.2f}%)")
    
    print("\n" + "=" * 70)
    print(f"Test set: {len(df)} matches")
    print("=" * 70)


if __name__ == "__main__":
    main()
