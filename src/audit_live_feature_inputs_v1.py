import sqlite3
from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

DATABASE_FILE = (
    PROJECT_ROOT
    /
    "data"
    /
    "cfb.db"
)

FEATURE_FILE = (
    PROJECT_ROOT
    /
    "data"
    /
    "processed"
    /
    "features_v2.csv"
)

FEATURE_BUILDER = (
    PROJECT_ROOT
    /
    "src"
    /
    "build_features_v1.py"
)


# ============================================================
# HELPERS
# ============================================================

def header(text):

    print()

    print(
        "=" * 110
    )

    print(
        text
    )

    print(
        "=" * 110
    )


def find_column(
    columns,
    candidates
):

    lower_map = {
        column.lower(): column
        for column in columns
    }


    for candidate in candidates:

        if candidate.lower() in lower_map:

            return lower_map[
                candidate.lower()
            ]


    return None


# ============================================================
# START
# ============================================================

print(
    "LIVE FEATURE INPUT AUDIT — V1"
)

print(
    "=" * 110
)


# ============================================================
# EXACT 54 MODEL FEATURES
# ============================================================

data = pd.read_csv(
    FEATURE_FILE,
    nrows=5
)


v1_columns = list(
    data.columns[
        :192
    ]
)


features = [

    column

    for column in v1_columns

    if column.startswith(
        "diff_pregame_"
    )
]


features += [

    "neutral_site",

    "pregame_elo_diff",

    "talent_diff",

    "returning_ppa_diff",

    "returning_passing_ppa_diff",

    "returning_receiving_ppa_diff",

    "returning_rushing_ppa_diff"
]


features = sorted(
    set(
        features
    )
)


header(
    "EXACT MODEL FEATURE CONTRACT"
)


print(
    f"Feature count: {len(features)}"
)


for index, feature in enumerate(
    features,
    start=1
):

    print(
        f"{index:2}. {feature}"
    )


if len(
    features
) != 54:

    raise RuntimeError(
        f"Expected 54 features, found "
        f"{len(features)}."
    )


# ============================================================
# ROLLING DIFFERENCE FEATURES
# ============================================================

rolling_features = [

    feature

    for feature in features

    if feature.startswith(
        "diff_pregame_"
    )
]


header(
    "ROLLING FEATURES"
)


print(
    f"Rolling differences: "
    f"{len(rolling_features)}"
)


for feature in rolling_features:

    print(
        feature
    )


# ============================================================
# DATABASE
# ============================================================

connection = sqlite3.connect(
    DATABASE_FILE
)


tables = {

    row[0]

    for row in connection.execute(
        """
        SELECT name

        FROM sqlite_master

        WHERE type = 'table'
        """
    ).fetchall()
}


# ============================================================
# TEAM TALENT / RETURNING PRODUCTION
# ============================================================

for table in [

    "team_talent",

    "returning_production"
]:

    header(
        table.upper()
    )


    if table not in tables:

        print(
            "TABLE MISSING"
        )

        continue


    schema = connection.execute(
        f"""
        PRAGMA table_info({table})
        """
    ).fetchall()


    columns = [
        row[1]
        for row in schema
    ]


    print(
        "Columns:"
    )


    for row in schema:

        print(
            f"  {row[1]:32} "
            f"{row[2]:12} "
            f"PK={row[5]}"
        )


    print()

    total_rows = connection.execute(
        f"""
        SELECT COUNT(*)

        FROM {table}
        """
    ).fetchone()[0]


    print(
        f"Total rows: "
        f"{total_rows:,}"
    )


    season_column = find_column(

        columns,

        [
            "season",
            "year"
        ]
    )


    team_column = find_column(

        columns,

        [
            "team",
            "school"
        ]
    )


    if season_column:

        print()

        print(
            "Rows by season:"
        )


        season_rows = connection.execute(
            f"""
            SELECT
                {season_column},
                COUNT(*)

            FROM {table}

            GROUP BY {season_column}

            ORDER BY {season_column}
            """
        ).fetchall()


        for season, count in season_rows:

            print(
                f"  {season}: "
                f"{count:,}"
            )


        rows_2026 = connection.execute(
            f"""
            SELECT COUNT(*)

            FROM {table}

            WHERE {season_column} = 2026
            """
        ).fetchone()[0]


        print()

        print(
            f"2026 rows: "
            f"{rows_2026:,}"
        )


    else:

        print()

        print(
            "No season/year column detected."
        )


    print()

    print(
        "Sample rows:"
    )


    samples = connection.execute(
        f"""
        SELECT *

        FROM {table}

        ORDER BY rowid DESC

        LIMIT 5
        """
    ).fetchall()


    for row in samples:

        print(
            row
        )


# ============================================================
# 2026 TEAM STAT CATEGORIES
# ============================================================

header(
    "2026 TEAM STAT CATEGORIES"
)


categories = connection.execute(
    """
    SELECT
        s.category,
        COUNT(*)

    FROM team_game_stat_values AS s

    INNER JOIN games AS g
        ON g.game_id = s.game_id

    WHERE g.season = 2026

    GROUP BY s.category

    ORDER BY s.category
    """
).fetchall()


for category, count in categories:

    print(
        f"{category:35} "
        f"{count:,}"
    )


# ============================================================
# REQUIRED HISTORICAL RAW METRICS
# ============================================================

required_metric_names = [

    "totalYards",

    "netPassingYards",

    "rushingYards",

    "turnovers",

    "firstDowns",

    "sacks",

    "TFL",

    "yardsPerPass",

    "yardsPerRushAttempt",

    "thirdDownPct"
]


header(
    "REQUIRED STAT CATEGORY AVAILABILITY"
)


category_names = {
    category
    for category, count
    in categories
}


for metric in required_metric_names:

    exact = (
        metric
        in category_names
    )


    partial_matches = [

        category

        for category in category_names

        if (
            metric.lower()
            in category.lower()

            or

            category.lower()
            in metric.lower()
        )
    ]


    if exact:

        status = "EXACT"

    elif partial_matches:

        status = (
            "SIMILAR: "
            +
            ", ".join(
                sorted(
                    partial_matches
                )
            )
        )

    else:

        status = "NOT FOUND"


    print(
        f"{metric:30} "
        f"{status}"
    )


# ============================================================
# HISTORICAL FEATURE BUILDER SOURCE BLOCKS
#
# Print the exact sections responsible for building the
# team-game history before rolling averages.
# ============================================================

header(
    "BUILD_FEATURES_V1 — RAW HISTORY SECTION"
)


with open(
    FEATURE_BUILDER,
    "r",
    encoding="utf-8"
) as file:

    source_lines = file.readlines()


# Lines are 1-indexed when shown to user.
#
# This range should capture:
# - team game rows
# - stat pivot
# - raw stat parsing
# - opponent joins
# - derived metrics
#
# We intentionally stop before the rolling-history section
# already audited.
start_line = 70

end_line = 490


for number in range(
    start_line,
    min(
        end_line,
        len(source_lines)
    ) + 1
):

    print(
        f"{number:4}: "
        f"{source_lines[number - 1].rstrip()}"
    )


# ============================================================
# UPCOMING 2026 CONTEXT COVERAGE
# ============================================================

header(
    "UPCOMING 2026 FBS-v-FBS"
)


upcoming_count = connection.execute(
    """
    SELECT COUNT(*)

    FROM games

    WHERE
        season = 2026
        AND (
            home_points IS NULL
            OR away_points IS NULL
        )
        AND LOWER(home_classification) = 'fbs'
        AND LOWER(away_classification) = 'fbs'
    """
).fetchone()[0]


print(
    f"Upcoming FBS-v-FBS games: "
    f"{upcoming_count:,}"
)


week_two = connection.execute(
    """
    SELECT COUNT(*)

    FROM games

    WHERE
        season = 2026
        AND week = 2
        AND (
            home_points IS NULL
            OR away_points IS NULL
        )
        AND LOWER(home_classification) = 'fbs'
        AND LOWER(away_classification) = 'fbs'
    """
).fetchone()[0]


print(
    f"Upcoming Week 2 FBS-v-FBS: "
    f"{week_two:,}"
)


# ============================================================
# CURRENT-SEASON HISTORY COUNTS
# ============================================================

header(
    "2026 FBS-v-FBS HISTORY AVAILABLE"
)


completed_fbs = connection.execute(
    """
    SELECT COUNT(*)

    FROM games

    WHERE
        season = 2026
        AND home_points IS NOT NULL
        AND away_points IS NOT NULL
        AND LOWER(home_classification) = 'fbs'
        AND LOWER(away_classification) = 'fbs'
    """
).fetchone()[0]


print(
    f"Completed FBS-v-FBS games: "
    f"{completed_fbs:,}"
)


teams_with_history = connection.execute(
    """
    SELECT COUNT(
        DISTINCT team
    )

    FROM (

        SELECT home_team AS team

        FROM games

        WHERE
            season = 2026
            AND home_points IS NOT NULL
            AND away_points IS NOT NULL
            AND LOWER(home_classification) = 'fbs'
            AND LOWER(away_classification) = 'fbs'

        UNION

        SELECT away_team AS team

        FROM games

        WHERE
            season = 2026
            AND home_points IS NOT NULL
            AND away_points IS NOT NULL
            AND LOWER(home_classification) = 'fbs'
            AND LOWER(away_classification) = 'fbs'
    )
    """
).fetchone()[0]


print(
    f"FBS teams with prior FBS history: "
    f"{teams_with_history:,}"
)


connection.close()


# ============================================================
# COMPLETE
# ============================================================

header(
    "AUDIT COMPLETE"
)


print(
    "No APIs were called."
)


print()

print(
    "If 2026 talent and returning-production rows "
    "are missing, those must be refreshed before "
    "live neural inference."
)


print()

print(
    "After that, we can build the dedicated upcoming-game "
    "feature generator using the exact 54-feature contract."
)