from __future__ import annotations

from typing import Dict


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
    ) -> Dict[str, float]:
        clv_proxy = FeatureEngineer.calculate_clv_proxy(target_odds, sharp_odds)
        sharp_prob = 1.0 / sharp_odds if sharp_odds > 1.0 else 0.0
        sharp_vig = FeatureEngineer.calculate_vig(sharp_market)

        is_home = selection == home_team
        sent_delta = (sentiment_home - sentiment_away) if is_home else (sentiment_away - sentiment_home)
        inj_delta = (injuries_away - injuries_home) if is_home else (injuries_home - injuries_away)

        return {
            "sharp_implied_prob": float(sharp_prob),
            "clv": float(clv_proxy),
            "sharp_vig": float(sharp_vig),
            "sentiment_delta": float(sent_delta),
            "injury_delta": float(inj_delta),
            "form_winrate_l5": float(form_winrate_l5),
            "form_games_l5": float(form_games_l5),
        }
