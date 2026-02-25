from src.integrations.apisports_fetcher import APISportsFetcher
from src.integrations.news_fetcher import NewsFetcher
from src.integrations.odds_fetcher import OddsFetcher
from src.integrations.weather_fetcher import OpenMeteoFetcher


def main():
    print("Step2 smoke setup OK")
    print("- APISportsFetcher loaded")
    print("- NewsFetcher loaded")
    print("- OddsFetcher loaded")
    print("- OpenMeteoFetcher loaded")


if __name__ == "__main__":
    main()
