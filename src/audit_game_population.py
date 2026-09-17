import sqlite3


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

database_file = "data/cfb.db"

start_year = 2018
end_year = 2025


# --------------------------------------------------
# CONNECT
# --------------------------------------------------

connection = sqlite3.connect(
    database_file
)

cursor = connection.cursor()


print("GAME POPULATION AUDIT")
print("=" * 80)


# --------------------------------------------------
# OVERALL CLASSIFICATION COUNTS
# --------------------------------------------------

print("\nSEASON POPULATION")
print("-" * 80)


cursor.execute("""
SELECT

    season,

    SUM(
        CASE
            WHEN home_points IS NOT NULL
             AND away_points IS NOT NULL
            THEN 1
            ELSE 0
        END
    ) AS played,

    SUM(
        CASE
            WHEN home_points IS NOT NULL
             AND away_points IS NOT NULL
             AND home_classification = 'fbs'
             AND away_classification = 'fbs'
            THEN 1
            ELSE 0
        END
    ) AS fbs_vs_fbs,

    SUM(
        CASE
            WHEN home_points IS NOT NULL
             AND away_points IS NOT NULL
             AND (
                    home_classification != 'fbs'
                 OR away_classification != 'fbs'
             )
            THEN 1
            ELSE 0
        END
    ) AS involves_non_fbs

FROM games

WHERE season BETWEEN ? AND ?

GROUP BY season

ORDER BY season
""", (
    start_year,
    end_year
))


season_rows = cursor.fetchall()


total_played = 0
total_fbs_vs_fbs = 0
total_other = 0


for (
    season,
    played,
    fbs_vs_fbs,
    involves_non_fbs
) in season_rows:

    total_played += played or 0
    total_fbs_vs_fbs += fbs_vs_fbs or 0
    total_other += involves_non_fbs or 0


    if played:

        fbs_percent = (
            fbs_vs_fbs
            / played
            * 100
        )

    else:

        fbs_percent = 0


    marker = ""

    if season == 2025:

        marker = "  <-- HOLDOUT"


    print(
        f"{season}: "
        f"{played:,} played | "
        f"{fbs_vs_fbs:,} FBS-v-FBS | "
        f"{involves_non_fbs:,} other | "
        f"{fbs_percent:.2f}% FBS-v-FBS"
        f"{marker}"
    )


# --------------------------------------------------
# NON-FBS CLASSIFICATIONS
# --------------------------------------------------

print("\n")
print("NON-FBS OPPONENT CLASSIFICATIONS")
print("-" * 80)


cursor.execute("""
SELECT

    classification,
    COUNT(*) AS games

FROM (

    SELECT
        away_classification AS classification

    FROM games

    WHERE
        season BETWEEN ? AND ?
        AND home_points IS NOT NULL
        AND away_points IS NOT NULL
        AND home_classification = 'fbs'
        AND away_classification != 'fbs'

    UNION ALL

    SELECT
        home_classification AS classification

    FROM games

    WHERE
        season BETWEEN ? AND ?
        AND home_points IS NOT NULL
        AND away_points IS NOT NULL
        AND away_classification = 'fbs'
        AND home_classification != 'fbs'
)

GROUP BY classification

ORDER BY games DESC
""", (
    start_year,
    end_year,
    start_year,
    end_year
))


classification_rows = cursor.fetchall()


if classification_rows:

    for classification, count in classification_rows:

        print(
            f"{str(classification):<20} "
            f"{count:,}"
        )

else:

    print(
        "No non-FBS opponents found."
    )


# --------------------------------------------------
# BETTING COVERAGE FOR FBS-V-FBS ONLY
# --------------------------------------------------

print("\n")
print("FBS-V-FBS BETTING COVERAGE")
print("-" * 80)


for year in range(
    start_year,
    end_year + 1
):

    cursor.execute("""
    SELECT COUNT(*)

    FROM games

    WHERE
        season = ?
        AND home_points IS NOT NULL
        AND away_points IS NOT NULL
        AND home_classification = 'fbs'
        AND away_classification = 'fbs'
    """, (year,))

    fbs_games = cursor.fetchone()[0]


    cursor.execute("""
    SELECT COUNT(DISTINCT g.game_id)

    FROM games AS g

    JOIN betting_lines AS b
        ON g.game_id = b.game_id

    WHERE
        g.season = ?
        AND g.home_points IS NOT NULL
        AND g.away_points IS NOT NULL
        AND g.home_classification = 'fbs'
        AND g.away_classification = 'fbs'
        AND b.spread IS NOT NULL
    """, (year,))

    games_with_spread = (
        cursor.fetchone()[0]
    )


    if fbs_games:

        coverage = (
            games_with_spread
            / fbs_games
            * 100
        )

    else:

        coverage = 0


    marker = ""

    if year == 2025:

        marker = "  <-- HOLDOUT"


    print(
        f"{year}: "
        f"{games_with_spread:,} / "
        f"{fbs_games:,} "
        f"({coverage:.2f}%)"
        f"{marker}"
    )


# --------------------------------------------------
# SAMPLE FBS-V-NON-FBS GAMES
# --------------------------------------------------

print("\n")
print("EXAMPLE FBS-V-NON-FBS GAMES")
print("-" * 80)


cursor.execute("""
SELECT

    season,
    week,

    away_team,
    away_classification,
    away_points,

    home_team,
    home_classification,
    home_points

FROM games

WHERE
    season BETWEEN ? AND ?
    AND home_points IS NOT NULL
    AND away_points IS NOT NULL
    AND (
           home_classification != 'fbs'
        OR away_classification != 'fbs'
    )

ORDER BY
    season DESC,
    week

LIMIT 20
""", (
    start_year,
    end_year
))


sample_rows = cursor.fetchall()


for row in sample_rows:

    print(
        f"{row[0]} Week {row[1]} | "
        f"{row[2]} ({row[3]}) {row[4]} @ "
        f"{row[5]} ({row[6]}) {row[7]}"
    )


# --------------------------------------------------
# TOTALS
# --------------------------------------------------

print("\n")
print("MODEL POPULATION SUMMARY")
print("=" * 80)

print(
    f"All played games:             "
    f"{total_played:,}"
)

print(
    f"FBS-v-FBS games:              "
    f"{total_fbs_vs_fbs:,}"
)

print(
    f"Games involving non-FBS:      "
    f"{total_other:,}"
)


if total_played:

    model_percent = (
        total_fbs_vs_fbs
        / total_played
        * 100
    )

else:

    model_percent = 0


print(
    f"FBS-v-FBS share:              "
    f"{model_percent:.2f}%"
)


# --------------------------------------------------
# CLOSE
# --------------------------------------------------

connection.close()