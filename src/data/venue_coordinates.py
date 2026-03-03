"""Stadium / venue coordinates for weather enrichment.

Maps home-team names to (latitude, longitude) tuples.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

# Major venues — extend as needed
_COORDS: Dict[str, Tuple[float, float]] = {
    # --- Bundesliga ---
    "Bayern Munich": (48.2188, 11.6247),
    "Borussia Dortmund": (51.4926, 7.4519),
    "RB Leipzig": (51.3459, 12.3495),
    "Bayer Leverkusen": (51.0383, 7.0020),
    "Eintracht Frankfurt": (50.0685, 8.6453),
    "VfB Stuttgart": (48.7924, 9.2320),
    "Wolfsburg": (52.4319, 10.8038),
    "Freiburg": (48.0217, 7.8294),
    "Hoffenheim": (49.2387, 8.8883),
    "Werder Bremen": (53.0664, 8.8376),
    "Union Berlin": (52.4573, 13.5681),
    "Augsburg": (48.3231, 10.8862),
    "Mainz": (49.9843, 8.2245),
    "Heidenheim": (48.6777, 10.1448),
    "Darmstadt": (49.8617, 8.6742),
    "Koln": (50.9335, 6.8753),
    "Monchengladbach": (51.1746, 6.3859),
    # --- EPL ---
    "Arsenal": (51.5549, -0.1084),
    "Manchester City": (53.4831, -2.2004),
    "Liverpool": (53.4308, -2.9609),
    "Manchester United": (53.4631, -2.2913),
    "Chelsea": (51.4817, -0.1910),
    "Tottenham": (51.6043, -0.0662),
    "Newcastle": (54.9756, -1.6217),
    "Brighton": (50.8617, -0.0838),
    "West Ham": (51.5387, 0.0166),
    "Aston Villa": (52.5092, -1.8847),
    "Wolverhampton": (52.5905, -2.1305),
    "Crystal Palace": (51.3983, -0.0855),
    "Everton": (53.4388, -2.9664),
    "Nottingham Forest": (52.9400, -1.1326),
    "Fulham": (51.4749, -0.2217),
    "Bournemouth": (50.7352, -1.8384),
    "Burnley": (53.7890, -2.2302),
    "Luton": (51.8842, -0.4316),
    "Sheffield United": (53.3702, -1.4710),
    "Brentford": (51.4907, -0.2886),
    # --- NBA ---
    "Los Angeles Lakers": (34.0430, -118.2673),
    "Golden State Warriors": (37.7680, -122.3878),
    "Boston Celtics": (42.3662, -71.0621),
    "Milwaukee Bucks": (43.0451, -87.9174),
    "Phoenix Suns": (33.4457, -112.0712),
    "Denver Nuggets": (39.7487, -105.0077),
    "Philadelphia 76ers": (39.9012, -75.1720),
    "Miami Heat": (25.7814, -80.1870),
    # --- Tennis ---
    "Roland Garros": (48.8469, 2.2490),
    "Wimbledon": (51.4341, -0.2143),
    "US Open": (40.7498, -73.8468),
    "Australian Open": (-37.8218, 144.9783),
}


def get_venue_coordinates(team_or_venue: str) -> Optional[Tuple[float, float]]:
    """Return (lat, lng) for a team/venue name, or None if unknown."""
    return _COORDS.get(team_or_venue)
