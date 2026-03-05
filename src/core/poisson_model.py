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
DEFAULT_RHO: float = -0.13             # Dixon-Coles low-score dependence parameter


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
        rho: Optional[float] = None,
    ) -> None:
        self.home_advantage = home_advantage
        self.league_avg_goals = league_avg_goals
        self.learning_rate = learning_rate
        if rho is None:
            from src.core.settings import settings
            self.rho = settings.poisson_rho
        else:
            self.rho = rho

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
    def _score_matrix(home_xg: float, away_xg: float, rho: float = 0.0) -> list[list[float]]:
        """Build a SCORE_RANGE x SCORE_RANGE matrix of P(home=i, away=j).

        Uses the Dixon-Coles (1997) adjustment for low-scoring dependence.
        When ``rho`` is 0, this reduces to the standard independent Poisson.

        The adjustment corrects for the empirical over-representation of
        low-scoring draws (0-0, 1-1) and under-representation of 1-0 / 0-1
        results that the naive independence assumption misses.

        Correction factors for low scores:
            P(0,0): *= 1 - home_xg * away_xg * rho
            P(1,0): *= 1 + away_xg * rho
            P(0,1): *= 1 + home_xg * rho
            P(1,1): *= 1 - rho
        """
        home_pmf = [poisson.pmf(i, home_xg) for i in range(SCORE_RANGE)]
        away_pmf = [poisson.pmf(j, away_xg) for j in range(SCORE_RANGE)]

        matrix = [
            [home_pmf[i] * away_pmf[j] for j in range(SCORE_RANGE)]
            for i in range(SCORE_RANGE)
        ]

        if rho != 0.0:
            # Dixon-Coles adjustment for low-scoring outcomes
            matrix[0][0] = max(0.0, matrix[0][0] * (1 - home_xg * away_xg * rho))
            matrix[1][0] = max(0.0, matrix[1][0] * (1 + away_xg * rho))
            matrix[0][1] = max(0.0, matrix[0][1] * (1 + home_xg * rho))
            matrix[1][1] = max(0.0, matrix[1][1] * (1 - rho))

            # Re-normalize so the matrix sums to 1.0
            total = sum(sum(row) for row in matrix)
            if total > 0:
                matrix = [[p / total for p in row] for row in matrix]

        return matrix

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
        matrix = self._score_matrix(home_xg, away_xg, rho=self.rho)

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
        """Bayesian Gamma-Poisson conjugate update for attack/defense strengths.

        Treats each team's attack and defense strength as the mean of a
        Gamma distribution (the conjugate prior for the Poisson rate).

        For a Gamma prior with shape α and rate β:
          - Prior mean = α / β
          - After observing k goals with expected rate λ:
            α_new = α + k
            β_new = β + λ/μ  (where μ = current strength, λ/μ = baseline rate)
          - Posterior mean = α_new / β_new

        The ``prior_weight`` (= 1 / learning_rate) controls how many
        "equivalent prior observations" the current strength represents.
        With lr=0.05, prior_weight=20, so a single match counts as ~5%
        of the evidence.

        This replaces the log-dampened heuristic which was numerically
        unstable for low-xG matches (e.g., xG=0.2, actual=3 → the old
        log1p approach could still produce 40%+ swings).  The Gamma
        conjugate naturally compresses outliers because the β denominator
        grows with the baseline rate.
        """
        home_xg, away_xg = self._expected_goals(home, away)

        h = self._load_strengths(home)
        a = self._load_strengths(away)

        # prior_weight = equivalent number of prior observations
        # With lr=0.05: prior_weight=20, so one match = 1/(20+1) ≈ 5% influence
        prior_weight = 1.0 / self.learning_rate

        def _gamma_update(current_strength: float, observed_goals: float, expected_goals: float) -> float:
            """Bayesian conjugate update: Gamma(α, β) posterior mean."""
            # α (shape) encodes prior total goals at this strength level
            alpha = prior_weight * current_strength
            # β (rate) encodes the number of expected-rate observations
            # We divide by max(current_strength, 0.01) to get the baseline rate
            baseline_rate = expected_goals / max(current_strength, 0.01)
            beta = prior_weight + baseline_rate
            # Posterior mean = (α + observed) / (β + 1 observation)
            return (alpha + observed_goals) / (beta + 1.0)

        # --- home team ---
        h["attack"] = _gamma_update(h["attack"], home_goals, home_xg)
        h["defense"] = _gamma_update(h["defense"], away_goals, away_xg)

        # --- away team ---
        a["attack"] = _gamma_update(a["attack"], away_goals, away_xg)
        a["defense"] = _gamma_update(a["defense"], home_goals, home_xg)

        # Clamp to sensible bounds so strengths don't run away
        for s in (h, a):
            s["attack"] = max(0.2, min(3.0, s["attack"]))
            s["defense"] = max(0.2, min(3.0, s["defense"]))

        self._save_strengths(home, h["attack"], h["defense"])
        self._save_strengths(away, a["attack"], a["defense"])

        log.debug(
            "Poisson strengths updated (Gamma conjugate): "
            "%s atk=%.3f def=%.3f | %s atk=%.3f def=%.3f",
            home, h["attack"], h["defense"],
            away, a["attack"], a["defense"],
        )

    # ----- helpers ----------------------------------------------------------

    def get_team_strengths(self, team: str) -> Dict[str, float]:
        """Return the current attack/defense ratings for *team*."""
        return self._load_strengths(team)
