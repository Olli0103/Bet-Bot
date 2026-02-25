import json
import math
import os


class QuantPricingModel:
    def __init__(self, weights_file="ml_strategy_weights.json"):
        self.weights = {
            "sentiment_delta": 0.0,
            "injury_delta": 0.0,
            "sharp_implied_prob": 1.0,
            "clv": 0.0,
            "sharp_vig": 0.0,
            "form_winrate_l5": 0.0,
            "form_games_l5": 0.0,
            "intercept": 0.0,
        }
        if os.path.exists(weights_file):
            try:
                with open(weights_file, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                for k, v in loaded.items():
                    if k in self.weights:
                        self.weights[k] = float(v)
            except Exception:
                pass

    def _log_odds_to_prob(self, log_odds: float) -> float:
        return 1.0 / (1.0 + math.exp(-max(-10, min(10, log_odds))))

    def get_true_probability(
        self,
        sharp_prob: float,
        sentiment: float = 0.0,
        injuries: float = 0.0,
        clv: float = 0.0,
        sharp_vig: float = 0.0,
        form_winrate_l5: float = 0.0,
        form_games_l5: float = 0.0,
    ) -> float:
        # backward-compatible aliases
        sharp_implied_prob = float(sharp_prob)
        sentiment_delta = float(sentiment)
        injury_delta = float(injuries)

        log_odds = float(self.weights.get("intercept", 0.0))
        log_odds += sharp_implied_prob * float(self.weights.get("sharp_implied_prob", 0.0))
        log_odds += sentiment_delta * float(self.weights.get("sentiment_delta", 0.0))
        log_odds += injury_delta * float(self.weights.get("injury_delta", 0.0))
        log_odds += float(clv) * float(self.weights.get("clv", 0.0))
        log_odds += float(sharp_vig) * float(self.weights.get("sharp_vig", 0.0))
        log_odds += float(form_winrate_l5) * float(self.weights.get("form_winrate_l5", 0.0))
        log_odds += float(form_games_l5) * float(self.weights.get("form_games_l5", 0.0))

        return self._log_odds_to_prob(log_odds)
