from bs4 import BeautifulSoup
import requests
import json
from dotenv import load_dotenv
from pymongo import MongoClient
import os
from constants import EXPECTED_TEAM_FIELDS, AP_TOP_25
import time

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

def main():
    load_dotenv()
    uri = os.getenv("MONGODB_URI")
    client = MongoClient(uri)

    db = client["march-madness"]
    container = db["team-stats"]

    invalid_teams = []

    for team in AP_TOP_25:
        url = f"https://www.sports-reference.com/cbb/schools/{team}/men/2026.html"
        try:
            response = requests.get(url)  # TODO: add user agent header to avoid

            if response.status_code != 200:
                invalid_teams.append(team)
                print(
                    f"{team}: Failed to retrieve data (Status code: {response.status_code})"
                )
            else:
                soup = BeautifulSoup(response.content, "html.parser")

                # <th scope="row" class="left " data-stat="entity">Team</th>
                total_team_table = soup.find("div", {"id": "div_season-total_totals"})
                team_row = total_team_table.find_all("tr")[1]
                season_stats_tds = team_row.find_all("td")

                stats = {"team": team}

                for td in season_stats_tds:
                    stat_name = td.get("data-stat")
                    stat_value = td.text.strip()

                    # Validate that we have both name and value
                    if stat_name and stat_value:
                        stats[stat_name] = convert_value(stat_value)

                missing_fields = [
                    field for field in EXPECTED_TEAM_FIELDS if field not in stats
                ]

                if missing_fields:
                    print(f"Warning: Missing fields: {missing_fields}")

                # TODO: convert numeric fields to int or float

                # print(json.dumps(stats, indent=2))

                container.insert_one(stats)

        except Exception as e:
            print(f"{team}: Exception - {e}")
            invalid_teams.append(team)

        time.sleep(2)  # Delay to avoid rate limiting (429)

    if invalid_teams:
        print(f"Warning: The following teams had invalid URLs: {invalid_teams}")


if __name__ == "__main__":
    main()
