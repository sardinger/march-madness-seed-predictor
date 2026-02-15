from bs4 import BeautifulSoup
import requests
import json
from dotenv import load_dotenv
from pymongo import MongoClient
import os
from constants import EXPECTED_ROLLING_FIELDS, AP_TOP_25
import time
from utils import get_game_rows


def main():
    load_dotenv()
    uri = os.getenv("MONGODB_URI")
    client = MongoClient(uri)

    db = client["march-madness"]
    container = db["rolling-stats"]

    invalid_teams = []

    for team in AP_TOP_25:
        url = f"https://www.sports-reference.com/cbb/schools/{team}/men/2026-schedule.html"
        try:
            response = requests.get(
                url
            )  # TODO: add user agent header to avoid rate limiting

            if response.status_code != 200:
                invalid_teams.append(team)
                print(
                    f"{team}: Failed to retrieve data (Status code: {response.status_code})"
                )
            else:
                soup = BeautifulSoup(response.content, "html.parser")

                results_table = soup.find("table", {"id": "schedule"})
                games = results_table.find_all("tr")[1:]  # skip the table header
                rolling_games = get_game_rows(games)

                # iterate thru each game and extract td stats
                for game in rolling_games:
                    game_stats = {"team": team}
                    tds = game.find_all("td")

                    for td in tds:
                        stat_name = td.get("data-stat")
                        stat_value = td.text.strip()

                        # Validate that we have both name and value
                        if stat_name and stat_value:
                            game_stats[stat_name] = stat_value

                    missing_fields = [
                        field
                        for field in EXPECTED_ROLLING_FIELDS
                        if field not in game_stats
                    ]

                    if missing_fields:
                        print(f"Warning: Missing fields: {missing_fields}")

                    # TODO: convert numeric fields to int or float

                    # print(json.dumps(stats, indent=2))

                    container.insert_one(game_stats)

        except Exception as e:
            print(f"{team}: Exception - {e}")
            invalid_teams.append(team)

        # time.sleep(2)  # Delay to avoid rate limiting (429)

    if invalid_teams:
        print(f"Warning: The following teams had invalid URLs: {invalid_teams}")


if __name__ == "__main__":
    main()
