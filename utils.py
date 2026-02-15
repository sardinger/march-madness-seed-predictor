def get_game_rows(games, n=10):
    """Helper function to get the game rows from the schedule page.

    Args:
        games (bs4.element): The result of finding all the game rows.
        n (int, optional): The number of game rows to return. Defaults to 10.
    Returns:
        list: A list of bs4.element objects representing the game rows."""

    game_rows = []
    for game in reversed(games):
        if len(game_rows) >= n:
            break

        game_result = game.find("td", {"data-stat": "game_result"})

        if not game_result:
            continue

        if game_result.text.strip() == "":
            continue

        game_rows.append(game)

    return game_rows
