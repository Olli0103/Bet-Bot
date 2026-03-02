from __future__ import annotations

from typing import Dict, List, Optional


class FeatureEngineer:
    @staticmethod
    def calculate_clv_proxy(target_odds: float, sharp_odds: float) -> float:
        """
        Relative edge proxy vs sharp odds.
        0.05 => target price is 5% better than sharp.
        """
        if sharp_odds <= 1.0 or target_odds <= 1.0:
            return 0.0
        return round((target_odds / sharp_odds) - 1.0, 4)

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
        twitter_sentiment_delta: float = 0.0,
        time_to_kickoff_hours: float = 24.0,
    ) -> Dict[str, float]:
        clv_proxy = FeatureEngineer.calculate_clv_proxy(target_odds, sharp_odds)
        sharp_prob = 1.0 / sharp_odds if sharp_odds > 1.0 else 0.0
        sharp_vig = FeatureEngineer.calculate_vig(sharp_market)

        is_home = selection == home_team
        sent_delta = (sentiment_home - sentiment_away) if is_home else (sentiment_away - sentiment_home)
        inj_delta = (injuries_away - injuries_home) if is_home else (injuries_home - injuries_away)

        features = {
            "sharp_implied_prob": float(sharp_prob),
            "clv": float(clv_proxy),
            "sharp_vig": float(sharp_vig),
            "sentiment_delta": float(sent_delta),
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
            "twitter_sentiment_delta": float(twitter_sentiment_delta),
            "time_to_kickoff_hours": float(time_to_kickoff_hours),
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
        clv_over = FeatureEngineer.calculate_clv_proxy(over_odds, sharp_over_odds)
        clv_under = FeatureEngineer.calculate_clv_proxy(under_odds, sharp_under_odds)
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
        clv = FeatureEngineer.calculate_clv_proxy(spread_odds, sharp_spread_odds)
        return {
            "spread_point": float(point),
            "sharp_spread_prob": round(sharp_prob, 4),
            "clv_spread": float(clv),
        }
