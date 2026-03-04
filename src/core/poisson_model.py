"""Poisson goal-distribution model for soccer match prediction.

Uses per-team attack/defense strength ratings (cached in Redis) and a score
matrix approach to derive 1X2, over/under 0.5/1.5/2.5/3.5, and BTTS probabilities.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

from scipy.stats import poisson

from src.core.sport_mapping import normalize_team as _normalize_team
from src.data.redis_cache import cache

log = logging.getLogger(__name__)

# ----- constants / defaults ------------------------------------------------

LEAGUE_AVG_GOALS: float = 1.35          # average goals per team per match
HOME_ADVANTAGE: float = 1.2             # multiplicative home boost
DEFAULT_ATTACK: float = 1.0             # league-average attack strength
DEFAULT_DEFENSE: float = 1.0            # league-average defense strength
SCORE_RANGE: int = 7                    # compute P(goals=0..6) per side
STRENGTH_TTL: int = 180 * 24 * 3600    # 180 days — survives the ~80-day summer break
LEARNING_RATE: float = 0.05            # how fast strengths move per result


def _cache_key(team: str) -> str:
    return f"poisson:team:{_normalize_team(team)}"


class PoissonSoccerModel:
    """Predict soccer match outcomes via independent Poisson goal distributions.

    Each team carries an *attack* and *defense* strength (default 1.0 = league
    average).  Strengths are persisted in Redis so they survive restarts and
    improve over time as results are fed back via ``update_strengths``.

    Parameters
    ----------
    home_advantage : float
        Multiplicative factor applied to the home team's expected goals.
        Default ``1.2``.
    league_avg_goals : float
        Baseline expected goals per team per match.  Default ``1.35``.
    learning_rate : float
        Controls how aggressively strengths update after each result.
        Default ``0.05``.
    """

    def __init__(
        self,
        home_advantage: float = HOME_ADVANTAGE,
        league_avg_goals: float = LEAGUE_AVG_GOALS,
        learning_rate: float = LEARNING_RATE,
    ) -> None:
        self.home_advantage = home_advantage
        self.league_avg_goals = league_avg_goals
        self.learning_rate = learning_rate

    # ----- strength persistence --------------------------------------------

    def _load_strengths(self, team: str) -> Dict[str, float]:
        """Return ``{"attack": float, "defense": float}`` from Redis or defaults."""
        key = _cache_key(team)
        data = cache.get_json(key)
        if data and isinstance(data, dict):
            return {
                "attack": float(data.get("attack", DEFAULT_ATTACK)),
                "defense": float(data.get("defense", DEFAULT_DEFENSE)),
            }
        return {"attack": DEFAULT_ATTACK, "defense": DEFAULT_DEFENSE}

    def _save_strengths(self, team: str, attack: float, defense: float) -> None:
        key = _cache_key(team)
        cache.set_json(key, {"attack": round(attack, 6), "defense": round(defense, 6)}, ttl_seconds=STRENGTH_TTL)

    # ----- expected goals --------------------------------------------------

    def _expected_goals(self, home: str, away: str) -> tuple[float, float]:
        """Compute expected goals for each side.

        home_xg = league_avg * home_attack * away_defense * home_advantage
        away_xg = league_avg * away_attack * home_defense
        """
        h = self._load_strengths(home)
        a = self._load_strengths(away)

        home_xg = self.league_avg_goals * h["attack"] * a["defense"] * self.home_advantage
        away_xg = self.league_avg_goals * a["attack"] * h["defense"]

        return home_xg, away_xg

    # ----- score matrix ----------------------------------------------------

    @staticmethod
    def _score_matrix(home_xg: float, away_xg: float) -> list[list[float]]:
        """Build a SCORE_RANGE x SCORE_RANGE matrix of P(home=i, away=j).

        Uses scipy Poisson PMF; goals for each team are assumed independent.
        """
        home_pmf = [poisson.pmf(i, home_xg) for i in range(SCORE_RANGE)]
        away_pmf = [poisson.pmf(j, away_xg) for j in range(SCORE_RANGE)]

        return [
            [home_pmf[i] * away_pmf[j] for j in range(SCORE_RANGE)]
            for i in range(SCORE_RANGE)
        ]

    # ----- public prediction -----------------------------------------------

    def predict_match(self, home: str, away: str) -> Dict[str, float]:
        """Return a dictionary of match probabilities and expected goals.

        Keys
        ----
        h2h_home, h2h_draw, h2h_away : float
            1X2 probabilities (sum to ~1.0).
        over_2_5, under_2_5 : float
            Over / under 2.5 total goals.
        btts_yes, btts_no : float
            Both-teams-to-score yes / no.
        home_xg, away_xg : float
            Poisson-expected goals for each side.
        """
        home_xg, away_xg = self._expected_goals(home, away)
        matrix = self._score_matrix(home_xg, away_xg)

        p_home = 0.0
        p_draw = 0.0
        p_away = 0.0
        p_over_0_5 = 0.0
        p_over_1_5 = 0.0
        p_over_2_5 = 0.0
        p_over_3_5 = 0.0
        p_btts = 0.0

        for i in range(SCORE_RANGE):
            for j in range(SCORE_RANGE):
                p = matrix[i][j]
                total = i + j
                if i > j:
                    p_home += p
                elif i == j:
                    p_draw += p
                else:
                    p_away += p

                if total > 0:
                    p_over_0_5 += p
                if total > 1:
                    p_over_1_5 += p
                if total > 2:
                    p_over_2_5 += p
                if total > 3:
                    p_over_3_5 += p

                if i >= 1 and j >= 1:
                    p_btts += p

        return {
            "h2h_home": round(p_home, 6),
            "h2h_draw": round(p_draw, 6),
            "h2h_away": round(p_away, 6),
            "over_0_5": round(p_over_0_5, 6),
            "under_0_5": round(1.0 - p_over_0_5, 6),
            "over_1_5": round(p_over_1_5, 6),
            "under_1_5": round(1.0 - p_over_1_5, 6),
            "over_2_5": round(p_over_2_5, 6),
            "under_2_5": round(1.0 - p_over_2_5, 6),
            "over_3_5": round(p_over_3_5, 6),
            "under_3_5": round(1.0 - p_over_3_5, 6),
            "btts_yes": round(p_btts, 6),
            "btts_no": round(1.0 - p_btts, 6),
            "home_xg": round(home_xg, 4),
            "away_xg": round(away_xg, 4),
        }

    # ----- online learning -------------------------------------------------

    def update_strengths(
        self, home: str, away: str, home_goals: int, away_goals: int
    ) -> None:
        """Nudge attack/defense strengths towards the observed result.

        The update rule compares actual goals to the model's expected goals
        and shifts the relevant strengths proportionally via a simple
        multiplicative adjustment:

            new_attack  = old_attack  * (1 + lr * (actual/expected - 1))
            new_defense = old_defense * (1 + lr * (conceded/expected - 1))

        This keeps strengths bounded and stable while allowing the model to
        slowly adapt to form changes.
        """
        home_xg, away_xg = self._expected_goals(home, away)
        lr = self.learning_rate

        h = self._load_strengths(home)
        a = self._load_strengths(away)

        # --- home team ---
        if home_xg > 0:
            ratio_home_scored = home_goals / home_xg
            h["attack"] *= 1.0 + lr * (ratio_home_scored - 1.0)
        if away_xg > 0:
            ratio_home_conceded = away_goals / away_xg
            h["defense"] *= 1.0 + lr * (ratio_home_conceded - 1.0)

        # --- away team ---
        if away_xg > 0:
            ratio_away_scored = away_goals / away_xg
            a["attack"] *= 1.0 + lr * (ratio_away_scored - 1.0)
        if home_xg > 0:
            ratio_away_conceded = home_goals / home_xg
            a["defense"] *= 1.0 + lr * (ratio_away_conceded - 1.0)

        # Clamp to sensible bounds so strengths don't run away
        for s in (h, a):
            s["attack"] = max(0.2, min(3.0, s["attack"]))
            s["defense"] = max(0.2, min(3.0, s["defense"]))

        self._save_strengths(home, h["attack"], h["defense"])
        self._save_strengths(away, a["attack"], a["defense"])

        log.debug(
            "Poisson strengths updated: %s atk=%.3f def=%.3f | %s atk=%.3f def=%.3f",
            home, h["attack"], h["defense"],
            away, a["attack"], a["defense"],
        )

    # ----- helpers ----------------------------------------------------------

    def get_team_strengths(self, team: str) -> Dict[str, float]:
        """Return the current attack/defense ratings for *team*."""
        return self._load_strengths(team)
