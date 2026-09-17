import sqlite3
import pandas as pd


# ==================================================
# SETTINGS
# ==================================================

database_file = "data/cfb.db"

game_ids = [
    401645328,  # Rice @ Army
    401641034,  # New Mexico State @ Sam Houston
    401644689,  # Kent State @ Miami (OH)
    401644780   # Eastern Michigan @ Western Michigan
]


# ==================================================
# CONNECT
# ==================================================

connection = sqlite3.connect(
    database_file
)


print("PPA GAP DETAIL INVESTIGATION")
print("=" * 110)


# ==================================================
# GAME INFORMATION
# ==================================================

game_query = """
SELECT

    game_id,
    season,
    week,
    start_date,

    away_team,
    away_points,

    home_team,
    home_points

FROM games

WHERE game_id = ?
"""


# ==================================================
# RUN EACH GAME
# ==================================================

for game_id in game_ids:

    game = pd.read_sql_query(
        game_query,
        connection,
        params=(game_id,)
    )


    print("\n\n")
    print("=" * 110)

    print(
        f"GAME ID: {game_id}"
    )

    print("=" * 110)


    if len(game) == 0:

        print(
            "Game not found."
        )

        continue


    print(
        game.to_string(
            index=False
        )
    )


    # --------------------------------------------------
    # TOTAL PLAY COUNTS
    # --------------------------------------------------

    summary_query = """
    SELECT

        COUNT(*) AS total_plays,

        SUM(
            CASE
                WHEN ppa IS NOT NULL
                THEN 1
                ELSE 0
            END
        ) AS ppa_plays,

        MIN(period) AS first_period,
        MAX(period) AS last_period,

        MIN(play_number) AS first_play_number,
        MAX(play_number) AS last_play_number

    FROM plays

    WHERE game_id = ?
    """


    summary = pd.read_sql_query(
        summary_query,
        connection,
        params=(game_id,)
    )


    print("\nPLAY SUMMARY")
    print("-" * 110)

    print(
        summary.to_string(
            index=False
        )
    )


    # --------------------------------------------------
    # OFFENSE / DEFENSE PAIRS
    #
    # IMPORTANT:
    # This includes ALL plays, not just PPA plays.
    # --------------------------------------------------

    team_pair_query = """
    SELECT

        offense,
        defense,

        COUNT(*) AS total_plays,

        SUM(
            CASE
                WHEN ppa IS NOT NULL
                THEN 1
                ELSE 0
            END
        ) AS ppa_plays

    FROM plays

    WHERE game_id = ?

    GROUP BY
        offense,
        defense

    ORDER BY
        total_plays DESC
    """


    team_pairs = pd.read_sql_query(
        team_pair_query,
        connection,
        params=(game_id,)
    )


    print("\nOFFENSE / DEFENSE LABELS — ALL PLAYS")
    print("-" * 110)

    print(
        team_pairs.to_string(
            index=False
        )
    )


    # --------------------------------------------------
    # PERIOD COVERAGE
    # --------------------------------------------------

    period_query = """
    SELECT

        period,

        COUNT(*) AS total_plays,

        SUM(
            CASE
                WHEN ppa IS NOT NULL
                THEN 1
                ELSE 0
            END
        ) AS ppa_plays

    FROM plays

    WHERE game_id = ?

    GROUP BY period

    ORDER BY period
    """


    periods = pd.read_sql_query(
        period_query,
        connection,
        params=(game_id,)
    )


    print("\nPLAYS BY PERIOD")
    print("-" * 110)

    print(
        periods.to_string(
            index=False
        )
    )


    # --------------------------------------------------
    # PLAY TYPES
    # --------------------------------------------------

    type_query = """
    SELECT

        play_type,

        COUNT(*) AS total_plays,

        SUM(
            CASE
                WHEN ppa IS NOT NULL
                THEN 1
                ELSE 0
            END
        ) AS ppa_plays

    FROM plays

    WHERE game_id = ?

    GROUP BY play_type

    ORDER BY
        total_plays DESC,
        play_type
    """


    play_types = pd.read_sql_query(
        type_query,
        connection,
        params=(game_id,)
    )


    print("\nPLAY TYPES")
    print("-" * 110)

    print(
        play_types.to_string(
            index=False
        )
    )


    # --------------------------------------------------
    # PPA BY TEAM LABEL
    # --------------------------------------------------

    ppa_team_query = """
    SELECT

        offense,
        defense,

        COUNT(*) AS ppa_plays,

        AVG(ppa) AS average_ppa,

        MIN(ppa) AS minimum_ppa,

        MAX(ppa) AS maximum_ppa

    FROM plays

    WHERE
        game_id = ?
        AND ppa IS NOT NULL

    GROUP BY
        offense,
        defense

    ORDER BY
        ppa_plays DESC
    """


    ppa_teams = pd.read_sql_query(
        ppa_team_query,
        connection,
        params=(game_id,)
    )


    print("\nPPA BY OFFENSE / DEFENSE LABEL")
    print("-" * 110)


    if len(
        ppa_teams
    ) > 0:

        print(
            ppa_teams.to_string(
                index=False
            )
        )

    else:

        print(
            "No PPA values recorded for this game."
        )


    # --------------------------------------------------
    # SAMPLE FIRST 20 PLAYS
    # --------------------------------------------------

    first_query = """
    SELECT

        period,

        clock_minutes,
        clock_seconds,

        offense,
        defense,

        down,
        distance,

        yards_gained,

        play_type,
        ppa,

        play_text

    FROM plays

    WHERE game_id = ?

    ORDER BY
        period,
        play_number

    LIMIT 20
    """


    first_plays = pd.read_sql_query(
        first_query,
        connection,
        params=(game_id,)
    )


    print("\nFIRST 20 RECORDED PLAYS")
    print("-" * 110)

    print(
        first_plays.to_string(
            index=False
        )
    )


    # --------------------------------------------------
    # SAMPLE LAST 20 PLAYS
    # --------------------------------------------------

    last_query = """
    SELECT

        period,

        clock_minutes,
        clock_seconds,

        offense,
        defense,

        down,
        distance,

        yards_gained,

        play_type,
        ppa,

        play_text

    FROM plays

    WHERE game_id = ?

    ORDER BY
        period DESC,
        play_number DESC

    LIMIT 20
    """


    last_plays = pd.read_sql_query(
        last_query,
        connection,
        params=(game_id,)
    )


    print("\nLAST 20 RECORDED PLAYS")
    print("-" * 110)

    print(
        last_plays.to_string(
            index=False
        )
    )


    # --------------------------------------------------
    # BOX-SCORE PLAY VOLUME
    #
    # Compare reported rushing/pass attempts against
    # the play-by-play volume.
    # --------------------------------------------------

    box_query = """
    SELECT

        team,
        category,
        stat_value

    FROM team_game_stat_values

    WHERE
        game_id = ?

        AND category IN (
            'rushingAttempts',
            'completionAttempts',
            'totalYards',
            'firstDowns'
        )

    ORDER BY
        team,
        category
    """


    box = pd.read_sql_query(
        box_query,
        connection,
        params=(game_id,)
    )


    print("\nBOX-SCORE VOLUME CHECK")
    print("-" * 110)

    print(
        box.to_string(
            index=False
        )
    )


# ==================================================
# CLOSE
# ==================================================

connection.close()