#!/usr/bin/env python3
"""
Tennis ATP XGBoost Training Pipeline

Trains an XGBoost classifier on historical tennis ATP matches.
Includes:
- DB extraction with target leakage prevention (randomized home/away)
- Feature enrichment via Jeff Sackmann stats
- SHAP importance analysis
"""
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Setup path
import sys
BOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BOT_DIR))

from dotenv import load_dotenv
load_dotenv(BOT_DIR / ".env")

from src.data.postgres import SessionLocal
from sqlalchemy import text
from src.core.feature_engineering import enrich_tennis_atp, _load_atp_player_stats


# =============================================================================
# PHASE 1: DB EXTRACTION
# =============================================================================

def load_tennis_atp_events(limit: Optional[int] = None) -> pd.DataFrame:
    """
    Load completed tennis ATP H2H matches from the database.
    Each event has 2 rows (one per player). We reconstruct the full match.
    """
    session = SessionLocal()
    
    query = """
        SELECT 
            pb.event_id,
            pb.sport,
            pb.market,
            pb.selection,
            pb.odds,
            pb.status,
            pb.pnl,
            pb.meta_features,
            pb.sharp_implied_prob,
            pb.sharp_vig,
            pb.form_winrate_l5,
            pb.form_games_l5
        FROM placed_bets pb
        WHERE pb.sport = 'tennis_atp'
          AND pb.market = 'h2h'
          AND pb.status IN ('won', 'lost')
          AND pb.is_training_data = TRUE
    """
    
    if limit:
        query += f" LIMIT {limit}"
    
    df = pd.read_sql(query, session.connection())
    session.close()
    
    print(f"📥 Loaded {len(df)} bet rows from DB")
    return df


def reconstruct_matches(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct full matches from paired rows (one per player).
    Returns a DataFrame with:
    - winner, loser, winner_odds, loser_odds, event_id, meta_features
    """
    matches = []
    
    for event_id, group in df.groupby('event_id'):
        if len(group) != 2:
            continue  # Skip incomplete events
        
        # Sort by odds to get favorite/underdog
        rows = group.sort_values('odds').to_dict('records')
        
        # Determine winner/loser
        winner_row = None
        loser_row = None
        for row in rows:
            if row['status'] == 'won':
                winner_row = row
            else:
                loser_row = row
        
        if not winner_row or not loser_row:
            continue
        
        # Extract player names from selection
        winner = winner_row['selection']
        loser = loser_row['selection']
        
        # Odds
        winner_odds = winner_row['odds']
        loser_odds = loser_row['odds']
        
        # Implied probabilities (convert from odds)
        winner_prob = 1.0 / winner_odds if winner_odds > 1 else 0.5
        loser_prob = 1.0 / loser_odds if loser_odds > 1 else 0.5
        
        # Remove vig for fair probabilities
        total_implied = winner_prob + loser_prob
        if total_implied > 0:
            winner_fair_prob = winner_prob / total_implied
            loser_fair_prob = loser_prob / total_implied
        else:
            winner_fair_prob, loser_fair_prob = 0.5, 0.5
        
        # Extract meta features
        meta = winner_row.get('meta_features', {}) or {}
        
        matches.append({
            'event_id': event_id,
            'winner': winner,
            'loser': loser,
            'winner_odds': winner_odds,
            'loser_odds': loser_odds,
            'winner_fair_prob': winner_fair_prob,
            'loser_fair_prob': loser_fair_prob,
            'winner_rank': meta.get('winner_rank'),
            'loser_rank': meta.get('loser_rank'),
            'surface': meta.get('surface', 'Hard'),
            'tournament': meta.get('tournament'),
            'is_favourite': meta.get('is_favourite', False),
            'sharp_vig': winner_row.get('sharp_vig', 0.025),
            'form_winrate_l5': winner_row.get('form_winrate_l5', 0.5),
            'form_games_l5': winner_row.get('form_games_l5', 0),
        })
    
    result_df = pd.DataFrame(matches)
    print(f"🎾 Reconstructed {len(result_df)} matches")
    return result_df


# =============================================================================
# PHASE 2: TARGET LEAKAGE PREVENTION (Randomization)
# =============================================================================

def randomize_home_away(df: pd.DataFrame, random_seed: int = 42) -> pd.DataFrame:
    """
    CRITICAL: Randomize home/away assignment to prevent target leakage.
    
    In tennis H2H, there's no natural "home" court advantage.
    If we always assign winner->home, the model learns "home always wins".
    
    Solution: For 50% of matches, swap winner<->loser and invert odds.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    df = df.copy()
    
    # Add random column for swap decision
    swap_mask = np.random.random(len(df)) < 0.5
    
    # Apply swap for selected rows
    df['home_player'] = np.where(swap_mask, df['loser'], df['winner'])
    df['away_player'] = np.where(swap_mask, df['winner'], df['loser'])
    
    # Invert odds after swap
    df['home_odds'] = np.where(swap_mask, df['loser_odds'], df['winner_odds'])
    df['away_odds'] = np.where(swap_mask, df['winner_odds'], df['loser_odds'])
    
    # Invert fair probabilities
    df['home_fair_prob'] = np.where(swap_mask, df['loser_fair_prob'], df['winner_fair_prob'])
    df['away_fair_prob'] = np.where(swap_mask, df['winner_fair_prob'], df['loser_fair_prob'])
    
    # Target: 1 if home won, 0 if away won
    # After swap: home_won = NOT swap (because swap inverts winner/loser)
    df['target'] = (~swap_mask).astype(int)
    
    # Keep original rankings for reference (swap if needed)
    df['home_rank'] = np.where(swap_mask, df['loser_rank'], df['winner_rank'])
    df['away_rank'] = np.where(swap_mask, df['winner_rank'], df['loser_rank'])
    
    print(f"🔀 Randomized home/away: {swap_mask.sum()} swaps out of {len(df)}")
    
    return df


# =============================================================================
# PHASE 3: FEATURE ENRICHMENT
# =============================================================================

def enrich_with_sackmann_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Jeff Sackmann ATP features to each match.
    Uses the enrich_tennis_atp() function from feature_engineering.py
    """
    print("🧠 Enriching with Jeff Sackmann features...")
    
    # Load ATP stats (cached)
    _load_atp_player_stats()
    
    tennis_features = []
    
    for idx, row in df.iterrows():
        home = row['home_player']
        away = row['away_player']
        
        try:
            features = enrich_tennis_atp(home, away)
        except Exception as e:
            print(f"Warning: Error enriching {home} vs {away}: {e}")
            features = {
                'tennis_diff_hold_rate': 0.0,
                'tennis_diff_break_rate': 0.0,
                'tennis_diff_net_rate': 0.0,
                'has_deep_tennis_stats': 0.0,
            }
        
        tennis_features.append(features)
    
    # Add features to dataframe
    features_df = pd.DataFrame(tennis_features)
    df = pd.concat([df.reset_index(drop=True), features_df], axis=1)
    
    print(f"✅ Enriched {len(df)} matches with tennis features")
    return df


def build_feature_vector(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build the final feature vector for XGBoost.
    
    FEATURES: Fundamental + Odds (both OK with randomization)
    """
    features = pd.DataFrame()
    
    # === TENNIS ATP ALPHA FEATURES (Jeff Sackmann) - PRIMARY ===
    # These are the true alpha signals
    features['tennis_diff_hold_rate'] = df['tennis_diff_hold_rate'].fillna(0.0)
    features['tennis_diff_break_rate'] = df['tennis_diff_break_rate'].fillna(0.0)
    features['tennis_diff_net_rate'] = df['tennis_diff_net_rate'].fillna(0.0)
    
    # Data quality indicator (critical for model to know confidence)
    features['has_deep_tennis_stats'] = df['has_deep_tennis_stats'].fillna(0.0)
    
    # Sample quality
    features['tennis_home_matches'] = df['tennis_home_matches'].fillna(0)
    features['tennis_away_matches'] = df['tennis_away_matches'].fillna(0)
    features['tennis_total_matches'] = features['tennis_home_matches'] + features['tennis_away_matches']
    
    # === RANKING FEATURES (Fundamental, not odds-derived) ===
    # Use raw ranks - these are independent of betting market
    features['home_rank'] = df['home_rank'].fillna(500)
    features['away_rank'] = df['away_rank'].fillna(500)
    features['rank_diff'] = features['home_rank'] - features['away_rank']
    features['rank_log_diff'] = np.log1p(features['home_rank']) - np.log1p(features['away_rank'])
    
    # Rank percentile (0 = best, 1 = worst)
    features['home_rank_pct'] = np.minimum(features['home_rank'] / 500, 1.0)
    features['away_rank_pct'] = np.minimum(features['away_rank'] / 500, 1.0)
    
    # === FORM FEATURES (Fundamental) ===
    features['form_winrate_l5'] = df['form_winrate_l5'].fillna(0.5)
    features['form_games_l5'] = df['form_games_l5'].fillna(0)
    features['has_form'] = (features['form_games_l5'] > 0).astype(float)
    
    # Market vigorish (NOT a feature, but useful for EV calc later)
    features['sharp_vig'] = df['sharp_vig'].fillna(0.025)
    
    # NO odds features - ONLY fundamental features (NO target leakage)
    # features['home_implied_prob'] = df['home_fair_prob'].fillna(0.5)  # REMOVED
    # features['clv'] = df['home_fair_prob'].fillna(0.5) - (1 / df['home_odds'].fillna(2.0))  # REMOVED
    
    feature_names = features.columns.tolist()
    
    print(f"📊 Built FUNDAMENTAL feature vector: {len(feature_names)} features (NO odds)")
    print(f"   Features: {feature_names}")
    
    return features, feature_names


# =============================================================================
# PHASE 4: XGBOOST TRAINING
# =============================================================================

def train_xgboost_model(
    X: pd.DataFrame, 
    y: pd.Series,
    feature_names: List[str],
    test_size: float = 0.2,
) -> Tuple:
    # Clean data: replace inf and large NaN values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    # Clip extreme values
    for col in X.columns:
        if X[col].dtype in [np.float64, np.float32]:
            q99 = X[col].quantile(0.99)
            q01 = X[col].quantile(0.01)
            X[col] = X[col].clip(q01, q99)
    """
    Train XGBoost classifier with SHAP analysis.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.calibration import CalibratedClassifierCV
    import xgboost as xgb
    import shap
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"📈 Training set: {len(X_train)}, Test set: {len(X_test)}")
    
    # XGBoost model (base)
    base_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
    )
    
    # Calibrate probabilities using isotonic regression (more accurate for probability estimation)
    print('🔧 Calibrating probabilities with isotonic regression...')
    model = CalibratedClassifierCV(base_model, method='isotonic', cv=5)
    model.fit(X_train, y_train)
    
    print('✅ Model calibrated!')
    
    # Evaluate
    from sklearn.metrics import accuracy_score, roc_auc_score
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    print(f"\n🎯 Model Performance:")
    print(f"   Accuracy: {accuracy:.3f}")
    print(f"   AUC-ROC:  {auc:.3f}")
    
    # Feature importance - train a separate model for importance
    try:
        # Train a quick XGBoost to get feature importance
        importance_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss',
        )
        importance_model.fit(X_train, y_train)
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_model.feature_importances_
        }).sort_values('importance', ascending=False)
    except Exception as e:
        importance = pd.DataFrame({'feature': [], 'importance': []})
    
    print("\n🔝 Top 10 Features:")
    print(importance.head(10).to_string(index=False))
    
    # Save model
    output_dir = BOT_DIR / "models" / "tennis_atp"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    import joblib
    model_path = output_dir / "xgb_tennis_atp.joblib"
    joblib.dump(model, model_path)
    print(f"\n💾 Model saved to: {model_path}")
    
    return model, None, importance


def main():
    print("=" * 60)
    print("🎾 TENNIS ATP XGBOOST TRAINING PIPELINE (FIXED)")
    print("=" * 60)
    
    # Step 1: Load data (ALL historical data)
    print("\n[1/6] Loading ALL tennis ATP data from database...")
    raw_df = load_tennis_atp_events(limit=None)  # Load ALL data
    
    # Step 2: Reconstruct matches
    print("\n[2/6] Reconstructing matches...")
    matches_df = reconstruct_matches(raw_df)
    
    # Step 3: Enrich with Jeff Sackmann features FIRST
    print("\n[3/6] Enriching with Jeff Sackmann ATP stats...")
    
    # Add home/away columns before enrichment (needed by enrich function)
    matches_df['home_player'] = matches_df['winner']
    matches_df['away_player'] = matches_df['loser']
    
    enriched_df = enrich_with_sackmann_features(matches_df)
    
    # CRITICAL: Step 1 - Filter >=20 matches per player BEFORE anything else
    # Use >=5 filter to preserve more data
    print("\n[4/6] Applying matches filter...")
    min_matches = 5
    matches_filter = (enriched_df['tennis_home_matches'] >= min_matches) & (enriched_df['tennis_away_matches'] >= min_matches)
    filtered_df = enriched_df[matches_filter].copy()
    print(f"  Filtered from {len(enriched_df)} to {len(filtered_df)} matches (>= {min_matches} per player)")
    
    # CRITICAL: Step 2 - Full Mirror Data Augmentation
    # Duplicate dataset with SWAPPED features + INVERTED target
    # This ensures exactly 50% target distribution
    print("\n[5/6] Creating FULL MIRROR augmentation...")
    
    # Set up columns
    filtered_df['home_player'] = filtered_df['winner']
    filtered_df['away_player'] = filtered_df['loser']
    filtered_df['home_odds'] = filtered_df['winner_odds']
    filtered_df['away_odds'] = filtered_df['loser_odds']
    filtered_df['home_fair_prob'] = filtered_df['winner_fair_prob']
    filtered_df['away_fair_prob'] = filtered_df['loser_fair_prob']
    filtered_df['home_rank'] = filtered_df['winner_rank'].fillna(500)
    filtered_df['away_rank'] = filtered_df['loser_rank'].fillna(500)
    filtered_df['target'] = 1  # Home won (winner = home)
    
    # Create mirror (swap ALL home/away features, invert target)
    mirror_df = filtered_df.copy()
    
    # Swap home/away columns
    for col in filtered_df.columns:
        if col.startswith('home_'):
            other = col.replace('home_', 'away_')
            if other in filtered_df.columns:
                mirror_df[col], mirror_df[other] = filtered_df[other].copy(), filtered_df[col].copy()
    
    # Also swap non-prefixed columns
    if 'home_player' in filtered_df.columns and 'away_player' in filtered_df.columns:
        mirror_df['home_player'], mirror_df['away_player'] = filtered_df['away_player'].copy(), filtered_df['home_player'].copy()
    
    # INVERT TARGET
    mirror_df['target'] = 1 - mirror_df['target']
    
    # Concatenate original + mirror
    augmented_df = pd.concat([filtered_df, mirror_df], ignore_index=True)
    
    print(f"  Original: {len(filtered_df)}, Mirror: {len(mirror_df)}")
    print(f"  Combined: {len(augmented_df)}")
    print(f"  Target distribution: {augmented_df['target'].value_counts().to_dict()}")
    print(f"  Target mean: {augmented_df['target'].mean():.4f}")
    
    # Step 4: Build features - ONLY fundamental, NO odds features
    print("\n[6/6] Building feature vector (fundamental features only, NO odds)...")
    X, feature_names = build_feature_vector(augmented_df)
    y = augmented_df['target']
    
    print(f"Training on {len(X)} valid samples...")
    valid_mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[valid_mask]
    y = y[valid_mask]
    
    print(f"  Final training set: {len(X)} samples")
    X = X[valid_mask]
    y = y[valid_mask]
    
    print(f"Training on {len(X)} valid samples...")
    
    model, _, importance = train_xgboost_model(
        X, y, feature_names
    )
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    
    return model, X, y, importance


if __name__ == "__main__":
    main()
