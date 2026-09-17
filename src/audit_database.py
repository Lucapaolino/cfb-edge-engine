import sqlite3


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

database_file = "data/cfb.db"
year = 2025


# --------------------------------------------------
# CONNECT
# --------------------------------------------------

connection = sqlite3.connect(database_file)

connection.execute("PRAGMA foreign_keys = ON")

cursor = connection.cursor()


print("2025 DATABASE INTEGRITY AUDIT")
print("=" * 80)


# --------------------------------------------------
# BASIC TABLE COUNTS
# --------------------------------------------------

tables = [
    "games",
    "team_game_stats",
    "team_game_stat_values",
    "plays",
    "team_talent",
    "returning_production",
    "betting_lines",
    "player_game_stat_values"
]

print("\nTABLE COUNTS")
print("-" * 80)

for table in tables:

    cursor.execute(
        f"""
        SELECT COUNT(*)
        FROM {table}
        """
    )

    count = cursor.fetchone()[0]

    print(
        f"{table:<30} "
        f"{count:,}"
    )


# --------------------------------------------------
# GAME COVERAGE
# --------------------------------------------------

print("\nGAME COVERAGE")
print("-" * 80)

cursor.execute("""
SELECT COUNT(*)
FROM games
WHERE season = ?
""", (year,))

total_games = cursor.fetchone()[0]

print(
    f"Games in season:                     "
    f"{total_games:,}"
)


# Team stats coverage
cursor.execute("""
SELECT COUNT(DISTINCT game_id)
FROM team_game_stats
WHERE game_id IN (
    SELECT game_id
    FROM games
    WHERE season = ?
)
""", (year,))

team_stat_games = cursor.fetchone()[0]

print(
    f"Games with team stats:                "
    f"{team_stat_games:,}"
)


# Play coverage
cursor.execute("""
SELECT COUNT(DISTINCT game_id)
FROM plays
WHERE game_id IN (
    SELECT game_id
    FROM games
    WHERE season = ?
)
""", (year,))

play_games = cursor.fetchone()[0]

print(
    f"Games with play-by-play:              "
    f"{play_games:,}"
)


# Betting coverage
cursor.execute("""
SELECT COUNT(DISTINCT game_id)
FROM betting_lines
WHERE game_id IN (
    SELECT game_id
    FROM games
    WHERE season = ?
)
""", (year,))

betting_games = cursor.fetchone()[0]

print(
    f"Games with betting lines:             "
    f"{betting_games:,}"
)


# Player stats coverage
cursor.execute("""
SELECT COUNT(DISTINCT game_id)
FROM player_game_stat_values
WHERE game_id IN (
    SELECT game_id
    FROM games
    WHERE season = ?
)
""", (year,))

player_stat_games = cursor.fetchone()[0]

print(
    f"Games with player stats:              "
    f"{player_stat_games:,}"
)


# --------------------------------------------------
# VERIFY EXACTLY TWO TEAM RECORDS PER GAME
# --------------------------------------------------

print("\nTEAM-GAME STRUCTURE")
print("-" * 80)

cursor.execute("""
SELECT COUNT(*)
FROM (
    SELECT
        game_id,
        COUNT(*) AS team_count

    FROM team_game_stats

    WHERE game_id IN (
        SELECT game_id
        FROM games
        WHERE season = ?
    )

    GROUP BY game_id

    HAVING COUNT(*) != 2
)
""", (year,))

bad_team_count_games = cursor.fetchone()[0]

print(
    f"Games without exactly 2 team rows:    "
    f"{bad_team_count_games:,}"
)


# --------------------------------------------------
# ORPHAN RECORD CHECKS
#
# A child row is "orphaned" if its game_id does not
# exist in the games table.
# --------------------------------------------------

print("\nFOREIGN-KEY / ORPHAN CHECKS")
print("-" * 80)

child_tables = [
    "team_game_stats",
    "team_game_stat_values",
    "plays",
    "betting_lines",
    "player_game_stat_values"
]

for table in child_tables:

    cursor.execute(
        f"""
        SELECT COUNT(*)

        FROM {table} AS child

        LEFT JOIN games AS g
            ON child.game_id = g.game_id

        WHERE g.game_id IS NULL
        """
    )

    orphan_count = cursor.fetchone()[0]

    print(
        f"{table:<30} "
        f"{orphan_count:,} orphan rows"
    )


# --------------------------------------------------
# FBS TEAM COUNT
# --------------------------------------------------

print("\nTEAM COVERAGE")
print("-" * 80)

cursor.execute("""
SELECT COUNT(*)
FROM (
    SELECT home_team AS team

    FROM games

    WHERE
        season = ?
        AND home_classification = 'fbs'

    UNION

    SELECT away_team AS team

    FROM games

    WHERE
        season = ?
        AND away_classification = 'fbs'
)
""", (year, year))

fbs_team_count = cursor.fetchone()[0]

print(
    f"Unique FBS teams found in games:      "
    f"{fbs_team_count:,}"
)


# --------------------------------------------------
# FIND FBS TEAMS WITHOUT TALENT DATA
# --------------------------------------------------

cursor.execute("""
WITH fbs_teams AS (

    SELECT home_team AS team
    FROM games
    WHERE
        season = ?
        AND home_classification = 'fbs'

    UNION

    SELECT away_team AS team
    FROM games
    WHERE
        season = ?
        AND away_classification = 'fbs'
)

SELECT f.team

FROM fbs_teams AS f

LEFT JOIN team_talent AS tt

    ON tt.season = ?
    AND tt.team = f.team

WHERE tt.team IS NULL

ORDER BY f.team
""", (year, year, year))

missing_talent = [
    row[0]
    for row in cursor.fetchall()
]

print(
    f"FBS teams missing talent data:        "
    f"{len(missing_talent)}"
)

if missing_talent:

    for team in missing_talent:
        print(
            f"  - {team}"
        )


# --------------------------------------------------
# FIND FBS TEAMS WITHOUT RETURNING PRODUCTION
# --------------------------------------------------

cursor.execute("""
WITH fbs_teams AS (

    SELECT home_team AS team
    FROM games
    WHERE
        season = ?
        AND home_classification = 'fbs'

    UNION

    SELECT away_team AS team
    FROM games
    WHERE
        season = ?
        AND away_classification = 'fbs'
)

SELECT f.team

FROM fbs_teams AS f

LEFT JOIN returning_production AS rp

    ON rp.season = ?
    AND rp.team = f.team

WHERE rp.team IS NULL

ORDER BY f.team
""", (year, year, year))

missing_returning = [
    row[0]
    for row in cursor.fetchall()
]

print(
    f"FBS teams missing returning data:     "
    f"{len(missing_returning)}"
)

if missing_returning:

    for team in missing_returning:
        print(
            f"  - {team}"
        )


# --------------------------------------------------
# PLAYER DATA SUMMARY
# --------------------------------------------------

print("\nPLAYER DATA")
print("-" * 80)

cursor.execute("""
SELECT COUNT(DISTINCT athlete_id)
FROM player_game_stat_values
""")

athletes = cursor.fetchone()[0]

print(
    f"Unique athletes:                      "
    f"{athletes:,}"
)


cursor.execute("""
SELECT COUNT(DISTINCT team)
FROM player_game_stat_values
""")

player_teams = cursor.fetchone()[0]

print(
    f"Teams represented in player stats:    "
    f"{player_teams:,}"
)


# --------------------------------------------------
# PLAY-BY-PLAY / PPA SUMMARY
# --------------------------------------------------

print("\nPLAY-BY-PLAY")
print("-" * 80)

cursor.execute("""
SELECT COUNT(*)
FROM plays
""")

total_plays = cursor.fetchone()[0]


cursor.execute("""
SELECT COUNT(*)
FROM plays
WHERE ppa IS NOT NULL
""")

ppa_plays = cursor.fetchone()[0]


if total_plays > 0:

    ppa_percent = (
        ppa_plays
        / total_plays
        * 100
    )

else:

    ppa_percent = 0


print(
    f"Total plays:                          "
    f"{total_plays:,}"
)

print(
    f"Plays with PPA:                       "
    f"{ppa_plays:,}"
)

print(
    f"Overall PPA coverage:                 "
    f"{ppa_percent:.2f}%"
)


# --------------------------------------------------
# FINAL PASS / WARNING SUMMARY
# --------------------------------------------------

print("\nAUDIT SUMMARY")
print("=" * 80)


problems = []


if total_games != 888:

    problems.append(
        f"Expected 888 games but found "
        f"{total_games}"
    )


if team_stat_games != total_games:

    problems.append(
        "Not every game has team stats"
    )


if play_games != total_games:

    problems.append(
        "Not every game has play-by-play"
    )


if betting_games != total_games:

    problems.append(
        "Not every game has betting data"
    )


if player_stat_games != total_games:

    problems.append(
        "Not every game has player stats"
    )


if bad_team_count_games != 0:

    problems.append(
        "Some games do not have exactly "
        "two team-game records"
    )


if problems:

    print("WARNINGS FOUND:")

    for problem in problems:

        print(
            f"  - {problem}"
        )

else:

    print(
        "Core 2025 game-level integrity "
        "checks passed."
    )


print(
    "\nMissing season-level context is "
    "reported separately above and does "
    "not automatically mean the database "
    "is corrupt."
)


# --------------------------------------------------
# CLOSE
# --------------------------------------------------

connection.close()