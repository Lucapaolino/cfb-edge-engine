import sqlite3


# --------------------------------------------------
# DATABASE LOCATION
# --------------------------------------------------

database_file = "data/cfb.db"


# --------------------------------------------------
# CONNECT TO DATABASE
# --------------------------------------------------

connection = sqlite3.connect(database_file)

cursor = connection.cursor()


# --------------------------------------------------
# QUERY
#
# We are combining:
#
# games
# +
# team_game_stats
# +
# team_game_stat_values
#
# using game_id and team.
# --------------------------------------------------

cursor.execute("""
SELECT

    g.game_id,
    g.season,
    g.week,
    g.start_date,

    g.away_team,
    g.away_points,

    g.home_team,
    g.home_points,

    t.team,
    t.home_away,

    s.category,
    s.stat_value

FROM games AS g

JOIN team_game_stats AS t
    ON g.game_id = t.game_id

JOIN team_game_stat_values AS s
    ON t.game_id = s.game_id
    AND t.team = s.team

WHERE t.team = 'Alabama'

AND s.category IN (
    'rushingYards',
    'netPassingYards',
    'totalYards',
    'turnovers',
    'thirdDownEff'
)

ORDER BY
    g.week,
    s.category
""")


rows = cursor.fetchall()


# --------------------------------------------------
# ORGANIZE RESULTS BY GAME
# --------------------------------------------------

games = {}

for row in rows:

    game_id = row[0]

    if game_id not in games:

        games[game_id] = {
            "season": row[1],
            "week": row[2],
            "start_date": row[3],
            "away_team": row[4],
            "away_points": row[5],
            "home_team": row[6],
            "home_points": row[7],
            "team": row[8],
            "home_away": row[9],
            "stats": {}
        }

    category = row[10]
    stat_value = row[11]

    games[game_id]["stats"][category] = stat_value


# --------------------------------------------------
# PRINT FIRST FIVE ALABAMA GAMES
# --------------------------------------------------

print("ALABAMA DATABASE JOIN TEST")
print("=" * 80)

for game_number, (game_id, game) in enumerate(
    games.items(),
    start=1
):

    if game_number > 5:
        break

    print(
        f"\n2025 Week {game['week']} | "
        f"{game['away_team']} {game['away_points']} @ "
        f"{game['home_team']} {game['home_points']}"
    )

    print(f"Game ID: {game_id}")
    print(f"Alabama was: {game['home_away']}")

    print("\nAlabama stats:")

    for category, value in game["stats"].items():

        print(
            f"  {category:<20}: {value}"
        )


# --------------------------------------------------
# CLOSE DATABASE
# --------------------------------------------------

connection.close()