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

connection = sqlite3.connect(database_file)

cursor = connection.cursor()


# --------------------------------------------------
# SEASON SUMMARY
# --------------------------------------------------

print("PLAYED VS UNPLAYED GAME AUDIT")
print("=" * 80)


cursor.execute("""
SELECT

    season,

    COUNT(*) AS total_records,

    SUM(
        CASE
            WHEN home_points IS NOT NULL
             AND away_points IS NOT NULL
            THEN 1
            ELSE 0
        END
    ) AS played_games,

    SUM(
        CASE
            WHEN home_points IS NULL
              OR away_points IS NULL
            THEN 1
            ELSE 0
        END
    ) AS unplayed_games

FROM games

WHERE season BETWEEN ? AND ?

GROUP BY season

ORDER BY season
""", (
    start_year,
    end_year
))


season_rows = cursor.fetchall()


print("\nSEASON COUNTS")
print("-" * 80)


total_records = 0
total_played = 0
total_unplayed = 0


for (
    season,
    records,
    played,
    unplayed
) in season_rows:

    total_records += records
    total_played += played
    total_unplayed += unplayed

    marker = ""

    if season == 2025:
        marker = "  <-- FINAL HOLDOUT"

    print(
        f"{season}: "
        f"{records:,} records | "
        f"{played:,} played | "
        f"{unplayed:,} unplayed"
        f"{marker}"
    )


# --------------------------------------------------
# LIST EVERY UNPLAYED GAME
# --------------------------------------------------

cursor.execute("""
SELECT

    season,
    week,
    game_id,
    start_date,

    away_team,
    away_classification,
    away_points,

    home_team,
    home_classification,
    home_points

FROM games

WHERE
    season BETWEEN ? AND ?
    AND (
        home_points IS NULL
        OR away_points IS NULL
    )

ORDER BY
    season,
    week,
    start_date
""", (
    start_year,
    end_year
))


unplayed_rows = cursor.fetchall()


print("\n")
print("UNPLAYED GAME RECORDS")
print("=" * 80)


if not unplayed_rows:

    print(
        "No unplayed games found."
    )

else:

    for row in unplayed_rows:

        print(
            f"\n{row[0]} Week {row[1]} | "
            f"Game ID {row[2]}"
        )

        print(
            f"  Date: {row[3]}"
        )

        print(
            f"  {row[4]} "
            f"({row[5]}) "
            f"{row[6]} "
            f"@ "
            f"{row[7]} "
            f"({row[8]}) "
            f"{row[9]}"
        )


# --------------------------------------------------
# MODEL-ELIGIBLE COUNT
#
# For now, our minimum requirement for a historical
# training example is that both teams have a final
# score.
# --------------------------------------------------

print("\n")
print("MODEL DATASET BASELINE")
print("=" * 80)

print(
    f"Total schedule records: "
    f"{total_records:,}"
)

print(
    f"Played games:           "
    f"{total_played:,}"
)

print(
    f"Unplayed games:         "
    f"{total_unplayed:,}"
)

print(
    f"\nMinimum model-eligible games "
    f"before other filters: "
    f"{total_played:,}"
)


# --------------------------------------------------
# CLOSE
# --------------------------------------------------

connection.close()