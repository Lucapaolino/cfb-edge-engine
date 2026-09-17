import os
import json
import sqlite3


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

year = 2024
database_file = "data/cfb.db"


# --------------------------------------------------
# CONNECT
# --------------------------------------------------

connection = sqlite3.connect(database_file)

cursor = connection.cursor()


# --------------------------------------------------
# FIND GAMES WITH NO TEAM STAT RECORDS
# --------------------------------------------------

cursor.execute("""
SELECT

    g.game_id,
    g.week,
    g.start_date,

    g.away_team,
    g.away_classification,
    g.away_conference,
    g.away_points,

    g.home_team,
    g.home_classification,
    g.home_conference,
    g.home_points

FROM games AS g

LEFT JOIN team_game_stats AS t
    ON g.game_id = t.game_id

WHERE
    g.season = ?
    AND t.game_id IS NULL

ORDER BY
    g.week,
    g.start_date
""", (year,))


missing_games = cursor.fetchall()


print("2024 MISSING TEAM STATS INVESTIGATION")
print("=" * 80)

print(
    f"Games missing team stats: "
    f"{len(missing_games)}"
)


# --------------------------------------------------
# INVESTIGATE EACH MISSING GAME
# --------------------------------------------------

for row in missing_games:

    game_id = row[0]
    week = row[1]
    start_date = row[2]

    away_team = row[3]
    away_classification = row[4]
    away_conference = row[5]
    away_points = row[6]

    home_team = row[7]
    home_classification = row[8]
    home_conference = row[9]
    home_points = row[10]


    print("\n" + "-" * 80)

    print(f"Game ID:     {game_id}")
    print(f"Week:        {week}")
    print(f"Start date:  {start_date}")

    print(
        f"\nAway: "
        f"{away_team} "
        f"({away_classification}, "
        f"{away_conference}) "
        f"- {away_points}"
    )

    print(
        f"Home: "
        f"{home_team} "
        f"({home_classification}, "
        f"{home_conference}) "
        f"- {home_points}"
    )


    # --------------------------------------------------
    # CHECK EXPECTED WEEK FILE
    # --------------------------------------------------

    raw_file = (
        f"data/raw/"
        f"team_stats_{year}_week{week}.json"
    )


    print(
        f"\nExpected raw file: "
        f"{raw_file}"
    )


    if not os.path.exists(raw_file):

        print(
            "ERROR: Raw week file does not exist."
        )

        continue


    with open(raw_file, "r") as file:

        raw_games = json.load(file)


    print(
        f"Games in raw week file: "
        f"{len(raw_games)}"
    )


    matching_raw_games = [
        game
        for game in raw_games
        if game.get("id") == game_id
    ]


    # --------------------------------------------------
    # GAME EXISTS IN RAW FILE
    # --------------------------------------------------

    if matching_raw_games:

        print(
            "\nRESULT: Game WAS found in "
            "the raw team-stat response."
        )

        print(
            "This points toward an import/database "
            "issue rather than missing source data."
        )

        raw_game = matching_raw_games[0]

        teams = raw_game.get(
            "teams",
            []
        )

        print(
            f"Teams in raw record: "
            f"{len(teams)}"
        )

        for team in teams:

            team_name = team.get("team")

            stats = team.get(
                "stats",
                []
            )

            print(
                f"  {team_name}: "
                f"{len(stats)} stat categories"
            )


    # --------------------------------------------------
    # GAME DOES NOT EXIST IN EXPECTED RAW FILE
    # --------------------------------------------------

    else:

        print(
            "\nRESULT: Game was NOT found in "
            "its expected raw week file."
        )


        # --------------------------------------------------
        # SEARCH ALL 2024 WEEK FILES
        #
        # This checks whether CFBD placed the game
        # under a different week than the /games
        # endpoint did.
        # --------------------------------------------------

        found_elsewhere = False


        for search_week in range(1, 17):

            search_file = (
                f"data/raw/"
                f"team_stats_{year}_"
                f"week{search_week}.json"
            )


            if not os.path.exists(
                search_file
            ):

                continue


            with open(
                search_file,
                "r"
            ) as file:

                search_games = json.load(
                    file
                )


            for raw_game in search_games:

                if raw_game.get("id") == game_id:

                    print(
                        f"\nFOUND in a different "
                        f"raw week file:"
                    )

                    print(
                        f"  Week {search_week}"
                    )

                    print(
                        f"  {search_file}"
                    )

                    found_elsewhere = True


        if not found_elsewhere:

            print(
                "\nGame ID was not found in ANY "
                "cached 2024 team-stat week."
            )

            print(
                "That strongly suggests the "
                "/games/teams source did not "
                "return a team box score for "
                "this game."
            )


# --------------------------------------------------
# SANITY CHECK
# --------------------------------------------------

cursor.execute("""
SELECT COUNT(*)
FROM games
WHERE season = ?
""", (year,))

game_count = cursor.fetchone()[0]


cursor.execute("""
SELECT COUNT(DISTINCT game_id)

FROM team_game_stats

WHERE game_id IN (

    SELECT game_id
    FROM games
    WHERE season = ?
)
""", (year,))

stats_game_count = cursor.fetchone()[0]


print("\n")
print("FINAL CHECK")
print("=" * 80)

print(
    f"2024 games:               "
    f"{game_count}"
)

print(
    f"Games with team stats:    "
    f"{stats_game_count}"
)

print(
    f"Difference:               "
    f"{game_count - stats_game_count}"
)


# --------------------------------------------------
# CLOSE
# --------------------------------------------------

connection.close()