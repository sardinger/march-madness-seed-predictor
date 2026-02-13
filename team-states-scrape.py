from bs4 import BeautifulSoup
import requests
import json


def main():
    url = "https://www.sports-reference.com/cbb/schools/michigan-state/men/2026.html#all_totals_team"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    # <th scope="row" class="left " data-stat="entity">Team</th>
    total_team_table = soup.find("div", {"id": "div_season-total_totals"})
    team_row = total_team_table.find_all("tr")[1]
    season_stats_tds = team_row.find_all("td")

    stats = {}

    for td in season_stats_tds:
        stat_name = td.get("data-stat")
        stat_value = td.text.strip()

        # Validate that we have both name and value
        if stat_name and stat_value:
            stats[stat_name] = stat_value

    expected_fields = [
        "games",
        "mp",
        "fg",
        "fga",
        "fg_pct",
        "fg2",
        "fg2a",
        "fg2_pct",
        "fg3",
        "fg3a",
        "fg3_pct",
        "ft",
        "fta",
        "ft_pct",
        "orb",
        "drb",
        "trb",
        "ast",
        "stl",
        "blk",
        "tov",
    ]

    missing_fields = [field for field in expected_fields if field not in stats]

    if missing_fields:
        print(f"Warning: Missing fields: {missing_fields}")

    # TODO: convert numeric fields to int or float

    print(json.dumps(stats, indent=2))

    return stats


if __name__ == "__main__":
    main()
