from typing import Dict, Any

from src.data.redis_cache import cache
from src.integrations.base_fetcher import AsyncBaseFetcher, _safe_sync_run


class OpenMeteoFetcher(AsyncBaseFetcher):
    def __init__(self):
        super().__init__(base_url="https://api.open-meteo.com/v1")

    async def get_hourly_weather_async(self, latitude: float, longitude: float, ttl_seconds: int = 900) -> Dict[str, Any]:
        cache_key = f"weather:{latitude}:{longitude}"
        cached = cache.get_json(cache_key)
        if cached:
            return cached

        data = await self.get(
            "forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "hourly": "temperature_2m,precipitation,wind_speed_10m",
                "timezone": "auto",
            },
        )
        cache.set_json(cache_key, data, ttl_seconds)
        return data

    def get_hourly_weather(self, latitude: float, longitude: float, ttl_seconds: int = 900) -> Dict[str, Any]:
        return _safe_sync_run(self.get_hourly_weather_async(latitude, longitude, ttl_seconds))
