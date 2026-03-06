from __future__ import annotations

from typing import Dict, List, Optional


def _to_float(value, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def calculate_smoothed_feature(
    measured_value: float,
    sample_size: int,
    prior_value: float,
    prior_weight: int = 10,
) -> float:
    """Apply Bayesian Smoothing to a noisy small-sample statistic.

    Shrinks the measured value towards a prior (e.g. league average or
    last season's closing rating) when the sample size is small.  This
    prevents wild early-season miscalibrations where a team that won
    its first 2 games gets an absurd 100% win rate.

    Formula (conjugate Beta-Binomial update for rates):
        smoothed = (measured * n + prior * prior_weight) / (n + prior_weight)

    Examples:
        - 2 wins in 2 games, prior=0.5, weight=10:
          (1.0*2 + 0.5*10) / (2+10) = 7/12 = 58.3%  (not 100%)
        - 20 wins in 30 games, prior=0.5, weight=10:
          (0.667*30 + 0.5*10) / (30+10) = 25/40 = 62.5%  (close to raw 66.7%)

    Parameters
    ----------
    measured_value : float
        Raw statistic (e.g. win rate 0.0-1.0, attack strength, etc.).
    sample_size : int
        Number of observations backing the measurement.
    prior_value : float
        Prior expectation (e.g. league average, last season's rating).
    prior_weight : int
        Effective number of "phantom observations" from the prior.
        Higher = more conservative smoothing.
    """
    measured_value = _to_float(measured_value, prior_value)
    sample_size = int(_to_float(sample_size, 0))
    prior_value = _to_float(prior_value, 0.0)
    prior_weight = int(_to_float(prior_weight, 10))

    if sample_size <= 0:
        return round(prior_value, 4)

    if sample_size + prior_weight == 0:
        return round(prior_value, 4)

    smoothed = (measured_value * sample_size + prior_value * prior_weight) / (
        sample_size + prior_weight
    )
    return round(smoothed, 4)


# Default prior weights for different feature categories
PRIOR_WEIGHTS = {
    "form_winrate": 5,       # 5 phantom games at league-average win rate
    "attack_strength": 10,   # 10 phantom games at league-average attack
    "defense_strength": 10,
    "over25_rate": 8,
    "btts_rate": 8,
    "goals_avg": 10,
}

# League-average priors (neutral baselines)
LEAGUE_PRIORS = {
    "form_winrate": 0.33,         # ~33% win rate (3-way market)
    "attack_strength": 1.0,       # Neutral Poisson attack
    "defense_strength": 1.0,      # Neutral Poisson defense
    "over25_rate": 0.50,          # 50% of matches go over 2.5
    "btts_rate": 0.50,            # 50% BTTS
    "goals_scored_avg": 1.35,     # ~1.35 goals per team per game
    "goals_conceded_avg": 1.35,
}


class FeatureEngineer:
    @staticmethod
    def calculate_clv_proxy(
        target_odds: float,
        sharp_odds: float,
        sharp_vig: float = 0.0,
    ) -> float:
        """Relative edge proxy vs the *fair* sharp price (vig removed).

        Raw sharp odds contain the bookmaker's margin (vig).  Comparing
        target_odds against vigged sharp odds systematically overstates
        the edge, which inflates Kelly stakes.

        Example: fair prob 50% → fair odds 2.00, Pinnacle 1.95 (2.6% vig).
        Tipico 2.05.  Without vig removal: 2.05/1.95-1 = +5.1%.
        With vig removal: 2.05/2.00-1 = +2.5% (correct).
        """
        if sharp_odds <= 1.0 or target_odds <= 1.0:
            return 0.0
        # Remove vig: convert sharp odds to fair odds.
        # Only apply correction when vig > 0 (valid multi-outcome market).
        # Negative or zero vig indicates incomplete market data — fall back
        # to raw sharp odds to avoid nonsensical corrections.
        raw_sharp_prob = 1.0 / sharp_odds
        if sharp_vig > 0:
            overround = 1.0 + sharp_vig  # e.g. 1.052 for 5.2% overround
            fair_sharp_prob = raw_sharp_prob / overround
        else:
            fair_sharp_prob = raw_sharp_prob
        fair_sharp_prob = max(fair_sharp_prob, 1e-9)  # avoid div-by-zero
        fair_sharp_odds = 1.0 / fair_sharp_prob
        return round((target_odds / fair_sharp_odds) - 1.0, 4)

    @staticmethod
    def calculate_vig(outcomes: Dict[str, float]) -> float:
        implied_probs = []
        for price in outcomes.values():
            if isinstance(price, (int, float)) and price > 1.0:
                implied_probs.append(1.0 / float(price))
        if not implied_probs:
            return 0.0
        return round(sum(implied_probs) - 1.0, 4)

    @staticmethod
    def market_consensus(all_books_prices: List[Dict[str, float]], selection: str) -> float:
        """Average implied probability for *selection* across all bookmakers."""
        probs = []
        for book_prices in all_books_prices:
            price = book_prices.get(selection)
            if price and price > 1.0:
                probs.append(1.0 / price)
        return round(sum(probs) / max(1, len(probs)), 4) if probs else 0.0

    @staticmethod
    def build_core_features(
        target_odds: float,
        sharp_odds: float,
        sharp_market: Dict[str, float],
        sentiment_home: float,
        sentiment_away: float,
        injuries_home: int,
        injuries_away: int,
        selection: str,
        home_team: str,
        form_winrate_l5: float = 0.5,
        form_games_l5: int = 0,
        elo_diff: float = 0.0,
        elo_expected: float = 0.5,
        h2h_home_winrate: float = 0.5,
        weather_rain: float = 0.0,
        weather_wind_high: float = 0.0,
        home_volatility: float = 0.0,
        away_volatility: float = 0.0,
        is_steam_move: bool = False,
        line_staleness: float = 0.0,
        poisson_true_prob: Optional[float] = None,
        injury_news_delta: float = 0.0,
        time_to_kickoff_hours: float = 24.0,
        public_bias: float = 0.0,
        public_hype_index: float = 0.0,
        market_momentum: float = 0.0,
        # Line movement velocity (implied prob change per hour)
        line_velocity: float = 0.0,
        # --- Phase 4: stats-based features (from EventStatsSnapshot) ---
        team_attack_strength: float = 1.0,
        team_defense_strength: float = 1.0,
        opp_attack_strength: float = 1.0,
        opp_defense_strength: float = 1.0,
        form_trend_slope: float = 0.0,
        rest_days: Optional[int] = None,
        schedule_congestion: float = 0.0,
        over25_rate: float = 0.0,
        btts_rate: float = 0.0,
        home_away_split_delta: float = 0.0,
        league_position_delta: float = 0.0,
        goals_scored_avg: float = 0.0,
        goals_conceded_avg: float = 0.0,
    ) -> Dict[str, float]:
        sharp_vig = FeatureEngineer.calculate_vig(sharp_market)
        clv_proxy = FeatureEngineer.calculate_clv_proxy(target_odds, sharp_odds, sharp_vig)

        # Strip vig (overround) from the sharp implied probability.
        # Raw 1/odds systematically overestimates the true probability by
        # the bookmaker's margin.  We normalise by dividing by the total
        # implied probability of all outcomes in the market.
        raw_sharp_prob = 1.0 / sharp_odds if sharp_odds > 1.0 else 0.0
        overround = 1.0 + sharp_vig  # e.g. 1.052 for a 5.2% overround
        sharp_prob = raw_sharp_prob / overround if overround > 0 else raw_sharp_prob

        is_home = selection == home_team
        sent_delta = (sentiment_home - sentiment_away) if is_home else (sentiment_away - sentiment_home)
        inj_delta = (injuries_away - injuries_home) if is_home else (injuries_home - injuries_away)

        # --- Bayesian Smoothing: shrink small-sample features to priors ---
        # Early-season (form_games_l5 < 5), raw stats are wildly unreliable.
        # Smoothing prevents a team with 2/2 wins from getting 100% win rate.
        n_games = max(0, int(_to_float(form_games_l5, 0)))
        if n_games > 0:
            form_winrate_l5 = calculate_smoothed_feature(
                form_winrate_l5, n_games,
                LEAGUE_PRIORS["form_winrate"], PRIOR_WEIGHTS["form_winrate"],
            )
            team_attack_strength = calculate_smoothed_feature(
                team_attack_strength, n_games,
                LEAGUE_PRIORS["attack_strength"], PRIOR_WEIGHTS["attack_strength"],
            )
            team_defense_strength = calculate_smoothed_feature(
                team_defense_strength, n_games,
                LEAGUE_PRIORS["defense_strength"], PRIOR_WEIGHTS["defense_strength"],
            )
            opp_attack_strength = calculate_smoothed_feature(
                opp_attack_strength, n_games,
                LEAGUE_PRIORS["attack_strength"], PRIOR_WEIGHTS["attack_strength"],
            )
            opp_defense_strength = calculate_smoothed_feature(
                opp_defense_strength, n_games,
                LEAGUE_PRIORS["defense_strength"], PRIOR_WEIGHTS["defense_strength"],
            )
            over25_rate = calculate_smoothed_feature(
                over25_rate, n_games,
                LEAGUE_PRIORS["over25_rate"], PRIOR_WEIGHTS["over25_rate"],
            )
            btts_rate = calculate_smoothed_feature(
                btts_rate, n_games,
                LEAGUE_PRIORS["btts_rate"], PRIOR_WEIGHTS["btts_rate"],
            )
            goals_scored_avg = calculate_smoothed_feature(
                goals_scored_avg, n_games,
                LEAGUE_PRIORS["goals_scored_avg"], PRIOR_WEIGHTS["goals_avg"],
            )
            goals_conceded_avg = calculate_smoothed_feature(
                goals_conceded_avg, n_games,
                LEAGUE_PRIORS["goals_conceded_avg"], PRIOR_WEIGHTS["goals_avg"],
            )

        # Expected total goals proxy: team_atk * opp_def * league_avg + opp_atk * team_def * league_avg
        expected_total_proxy = (team_attack_strength * opp_defense_strength * 1.35 +
                                opp_attack_strength * team_defense_strength * 1.35)

        # Rest fatigue score: 0 = well rested, 1 = congested
        rest_fatigue = 0.0
        rest_days_val = _to_float(rest_days, -1.0) if rest_days is not None else -1.0
        if rest_days_val >= 0:
            if rest_days_val <= 2:
                rest_fatigue = 1.0
            elif rest_days_val <= 4:
                rest_fatigue = 0.5
            elif rest_days_val >= 10:
                rest_fatigue = 0.3  # rustiness penalty

        # NLP placeholders (safe cold-start defaults for historical rows)
        public_hype = float(_to_float(public_hype_index, 0.0))
        smart_money_divergence = float(sent_delta) - public_hype

        features = {
            "sharp_implied_prob": float(sharp_prob),
            "clv": float(clv_proxy),
            "sharp_vig": float(sharp_vig),
            "sentiment_delta": float(sent_delta),
            "public_hype_index": public_hype,
            "smart_money_divergence": float(smart_money_divergence),
            "injury_delta": float(inj_delta),
            "form_winrate_l5": float(form_winrate_l5),
            "form_games_l5": float(form_games_l5),
            # Phase 2 features
            "elo_diff": float(elo_diff),
            "elo_expected": float(elo_expected),
            "h2h_home_winrate": float(h2h_home_winrate),
            "home_advantage": 1.0 if is_home else 0.0,
            "weather_rain": float(weather_rain),
            "weather_wind_high": float(weather_wind_high),
            "home_volatility": float(home_volatility),
            "away_volatility": float(away_volatility),
            "is_steam_move": 1.0 if is_steam_move else 0.0,
            "line_staleness": float(line_staleness),
            "injury_news_delta": float(injury_news_delta),
            "time_to_kickoff_hours": float(time_to_kickoff_hours),
            "public_bias": float(public_bias),
            "market_momentum": float(market_momentum),
            "line_velocity": float(line_velocity),
            # Phase 4: stats-based features
            "team_attack_strength": float(team_attack_strength),
            "team_defense_strength": float(team_defense_strength),
            "opp_attack_strength": float(opp_attack_strength),
            "opp_defense_strength": float(opp_defense_strength),
            "expected_total_proxy": round(expected_total_proxy, 4),
            "form_trend_slope": float(form_trend_slope),
            "rest_fatigue_score": float(rest_fatigue),
            "schedule_congestion": float(schedule_congestion),
            "over25_rate": float(over25_rate),
            "btts_rate": float(btts_rate),
            "home_away_split_delta": float(home_away_split_delta),
            "league_position_delta": float(league_position_delta),
            "goals_scored_avg": float(goals_scored_avg),
            "goals_conceded_avg": float(goals_conceded_avg),
        }
        if poisson_true_prob is not None:
            features["poisson_true_prob"] = float(poisson_true_prob)
        return features

    @staticmethod
    def build_totals_features(
        over_odds: float,
        under_odds: float,
        sharp_over_odds: float,
        sharp_under_odds: float,
        point: float,
        poisson_over_prob: Optional[float] = None,
    ) -> Dict[str, float]:
        """Features for over/under totals markets."""
        over_prob = 1.0 / sharp_over_odds if sharp_over_odds > 1.0 else 0.5
        under_prob = 1.0 / sharp_under_odds if sharp_under_odds > 1.0 else 0.5
        # Compute vig from the two-way market for vig-corrected CLV
        totals_vig = FeatureEngineer.calculate_vig(
            {"over": sharp_over_odds, "under": sharp_under_odds}
        )
        clv_over = FeatureEngineer.calculate_clv_proxy(over_odds, sharp_over_odds, totals_vig)
        clv_under = FeatureEngineer.calculate_clv_proxy(under_odds, sharp_under_odds, totals_vig)
        features = {
            "totals_point": float(point),
            "sharp_over_prob": round(over_prob, 4),
            "sharp_under_prob": round(under_prob, 4),
            "clv_over": float(clv_over),
            "clv_under": float(clv_under),
        }
        if poisson_over_prob is not None:
            features["poisson_over_prob"] = float(poisson_over_prob)
        return features

    @staticmethod
    def build_spreads_features(
        spread_odds: float,
        sharp_spread_odds: float,
        point: float,
    ) -> Dict[str, float]:
        """Features for spread/handicap markets."""
        sharp_prob = 1.0 / sharp_spread_odds if sharp_spread_odds > 1.0 else 0.5
        clv = FeatureEngineer.calculate_clv_proxy(spread_odds, sharp_spread_odds, 0.0)
        return {
            "spread_point": float(point),
            "sharp_spread_prob": round(sharp_prob, 4),
            "clv_spread": float(clv),
        }
