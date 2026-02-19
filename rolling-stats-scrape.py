from bs4 import BeautifulSoup
import requests
import json
from dotenv import load_dotenv
from pymongo import MongoClient
import os
from constants import EXPECTED_ROLLING_FIELDS, AP_TOP_25, CONFERENCE_MAP
import time
from utils import get_game_rows


# Changes all strings to ints or floats depending on what type of data is stored
def convert_value(value):
    if value in ["", "-", "NA", None]:
        return None
    
    # remove commas (attendance numbers)
    value = value.replace(",", "")
    
    try:
        return int(value)
    except:
        pass
    
    try:
        return float(value)
    except:
        pass
    
    return value

# Changes game_rsult from "W" -> 1, and "L" -> 0
def convert_game_result(value):
    if not value:
        return None
    
    v = value.strip().upper()
    
    if v.startswith("W"):
        return 1
    if v.startswith("L"):
        return 0
    
    return None

# parse to separate game streak and game streak result
def parse_game_streak(value):
    if not value:
        return None, None
    
    parts = value.strip().split()

    if len(parts) != 2:
        return None, None

    result, num = parts

    # W → 1, L → 0
    if result.upper().startswith("W"):
        streak_result = 1
    elif result.upper().startswith("L"):
        streak_result = 0
    else:
        streak_result = None

    try:
        streak_num = int(num)
    except:
        streak_num = None

    return streak_result, streak_num



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
                            if stat_name == "conf_abbr":
                                game_stats[stat_name] = CONFERENCE_MAP.get(stat_value, None)
                            # Map to conferences
                            #"Big Ten" = 1 "Big 12" = 2 "ACC" = 3 "SEC" = 4 "Big East" = 5 "WCC" = 6 "A-10" = 7 "MAC" = 8 "Sun Belt"= 9

                            # Change Game result to 1 or 0
                            elif stat_name == "game_result":
                                game_stats[stat_name] = convert_game_result(stat_value)


                            elif stat_name == "game_streak":
                                result, num = parse_game_streak(stat_value)
                                game_stats["game_streak_result"] = result
                                game_stats["game_streak_num"] = num
                            else:
                                game_stats[stat_name] = convert_value(stat_value)


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
                    # print(game_stats)

        except Exception as e:
            print(f"{team}: Exception - {e}")
            invalid_teams.append(team)

        time.sleep(2)  # Delay to avoid rate limiting (429)

    if invalid_teams:
        print(f"Warning: The following teams had invalid URLs: {invalid_teams}")




if __name__ == "__main__":
    main()
