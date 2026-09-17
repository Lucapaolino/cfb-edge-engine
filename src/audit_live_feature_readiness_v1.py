import re
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
    "features_v4.csv"
)

TRAIN_SCRIPT = (
    PROJECT_ROOT
    /
    "src"
    /
    "train_neural_network_v1.py"
)

BUILD_V1_SCRIPT = (
    PROJECT_ROOT
    /
    "src"
    /
    "build_features_v1.py"
)

QB_SCRIPT = (
    PROJECT_ROOT
    /
    "src"
    /
    "build_qb_continuity_features_v1.py"
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


def print_matching_lines(
    path,
    patterns,
    title
):

    header(
        title
    )


    if not path.exists():

        print(
            f"FILE NOT FOUND: {path}"
        )

        return


    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        lines = file.readlines()


    matches = []


    for line_number, line in enumerate(
        lines,
        start=1
    ):

        lower = line.lower()


        if any(
            pattern.lower() in lower
            for pattern in patterns
        ):

            matches.append(
                (
                    line_number,
                    line.rstrip()
                )
            )


    if not matches:

        print(
            "No matching lines found."
        )

        return


    for line_number, line in matches:

        print(
            f"{line_number:5}: {line}"
        )


# ============================================================
# START
# ============================================================

print(
    "LIVE FEATURE READINESS AUDIT — V1"
)

print(
    "=" * 110
)


# ============================================================
# FEATURE FILE
# ============================================================

if not FEATURE_FILE.exists():

    raise RuntimeError(
        f"Feature file not found: "
        f"{FEATURE_FILE}"
    )


features = pd.read_csv(
    FEATURE_FILE
)


header(
    "FEATURES_V4 DATASET"
)


print(
    f"Rows:    {len(features):,}"
)


print(
    f"Columns: {len(features.columns):,}"
)


if "season" in features.columns:

    seasons = sorted(

        pd.to_numeric(
            features[
                "season"
            ],
            errors="coerce"
        )

        .dropna()

        .astype(
            int
        )

        .unique()
    )


    print(
        f"Seasons: {seasons}"
    )


# ============================================================
# ALL COLUMN GROUPS
# ============================================================

column_groups = {

    "Pregame difference features":
        [
            column
            for column in features.columns
            if column.startswith(
                "diff_pregame_"
            )
        ],

    "Home pregame features":
        [
            column
            for column in features.columns
            if column.startswith(
                "home_pregame_"
            )
        ],

    "Away pregame features":
        [
            column
            for column in features.columns
            if column.startswith(
                "away_pregame_"
            )
        ],

    "Elo-related":
        [
            column
            for column in features.columns
            if "elo" in column.lower()
        ],

    "Talent-related":
        [
            column
            for column in features.columns
            if "talent" in column.lower()
        ],

    "Returning-production-related":
        [
            column
            for column in features.columns
            if "return" in column.lower()
        ],

    "QB-related":
        [
            column
            for column in features.columns
            if "qb" in column.lower()
        ],

    "PPA-related":
        [
            column
            for column in features.columns
            if "ppa" in column.lower()
        ]
}


for title, columns in column_groups.items():

    header(
        title.upper()
    )


    print(
        f"Count: {len(columns):,}"
    )


    for column in columns:

        print(
            f"  {column}"
        )


# ============================================================
# LIKELY ID / TARGET COLUMNS
# ============================================================

header(
    "IDENTIFIER / TARGET-LIKE COLUMNS"
)


keywords = [

    "game",
    "season",
    "week",
    "team",
    "point",
    "score",
    "margin",
    "winner",
    "win",
    "spread",
    "moneyline",
    "neutral"
]


for column in features.columns:

    lower = column.lower()


    if any(
        keyword in lower
        for keyword in keywords
    ):

        print(
            column
        )


# ============================================================
# SAMPLE HISTORICAL ROW
#
# We want to see a real row close to the current situation:
# early-season pregame values.
# ============================================================

header(
    "EARLY-SEASON HISTORICAL SAMPLE"
)


sample = features.copy()


if (
    "season" in sample.columns
    and
    "week" in sample.columns
):

    sample = sample[

        (
            pd.to_numeric(
                sample[
                    "season"
                ],
                errors="coerce"
            )
            ==
            2025
        )

        &

        (
            pd.to_numeric(
                sample[
                    "week"
                ],
                errors="coerce"
            )
            <=
            2
        )

    ]


    print(
        f"2025 Week 1-2 rows: "
        f"{len(sample):,}"
    )


    if len(
        sample
    ) > 0:

        row = sample.iloc[
            0
        ]


        identification_columns = [

            column

            for column in [

                "game_id",
                "season",
                "week",
                "away_team",
                "home_team",
                "neutral_site"

            ]

            if column in features.columns
        ]


        print()

        print(
            "Example game:"
        )


        for column in identification_columns:

            print(
                f"  {column}: "
                f"{row[column]}"
            )


        pregame_columns = [

            column

            for column in features.columns

            if (
                column.startswith(
                    "diff_pregame_"
                )
                or
                "elo" in column.lower()
                or
                "talent" in column.lower()
                or
                "return" in column.lower()
                or
                "qb" in column.lower()
            )
        ]


        non_null = row[
            pregame_columns
        ].notna().sum()


        null = row[
            pregame_columns
        ].isna().sum()


        print()

        print(
            f"Pregame/context columns checked: "
            f"{len(pregame_columns):,}"
        )


        print(
            f"Non-null: {non_null:,}"
        )


        print(
            f"Null:     {null:,}"
        )


# ============================================================
# 2025 FEATURE COMPLETENESS BY WEEK
# ============================================================

if (
    "season" in features.columns
    and
    "week" in features.columns
):

    header(
        "2025 FEATURE ROWS BY WEEK"
    )


    rows_2025 = features[

        pd.to_numeric(
            features[
                "season"
            ],
            errors="coerce"
        )
        ==
        2025
    ].copy()


    week_counts = (

        rows_2025

        .groupby(
            "week"
        )

        .size()
    )


    for week, count in week_counts.items():

        print(
            f"Week {week}: "
            f"{count:,} rows"
        )


# ============================================================
# DATABASE LIVE STATUS
# ============================================================

header(
    "LIVE 2026 DATABASE STATUS"
)


connection = sqlite3.connect(
    DATABASE_FILE
)


game_status = connection.execute(
    """
    SELECT

        COUNT(*) AS total,

        SUM(
            CASE
                WHEN home_points IS NOT NULL
                AND away_points IS NOT NULL
                THEN 1
                ELSE 0
            END
        ) AS completed,

        SUM(
            CASE
                WHEN home_points IS NULL
                OR away_points IS NULL
                THEN 1
                ELSE 0
            END
        ) AS upcoming

    FROM games

    WHERE season = 2026
    """
).fetchone()


print(
    f"2026 games:     "
    f"{game_status[0]:,}"
)


print(
    f"Completed:      "
    f"{game_status[1]:,}"
)


print(
    f"Upcoming:       "
    f"{game_status[2]:,}"
)


team_stat_games = connection.execute(
    """
    SELECT COUNT(
        DISTINCT g.game_id
    )

    FROM games AS g

    INNER JOIN team_game_stat_values AS s
        ON s.game_id = g.game_id

    WHERE g.season = 2026
    """
).fetchone()[0]


player_stat_games = connection.execute(
    """
    SELECT COUNT(
        DISTINCT g.game_id
    )

    FROM games AS g

    INNER JOIN player_game_stat_values AS p
        ON p.game_id = g.game_id

    WHERE g.season = 2026
    """
).fetchone()[0]


print(
    f"Games with team stats:   "
    f"{team_stat_games:,}"
)


print(
    f"Games with player stats: "
    f"{player_stat_games:,}"
)


# ============================================================
# UPCOMING FBS-FBS SAMPLE
# ============================================================

header(
    "UPCOMING FBS-v-FBS SAMPLE"
)


upcoming = connection.execute(
    """
    SELECT
        game_id,
        week,
        start_date,
        away_team,
        home_team,
        away_pregame_elo,
        home_pregame_elo,
        neutral_site

    FROM games

    WHERE
        season = 2026
        AND (
            home_points IS NULL
            OR away_points IS NULL
        )
        AND LOWER(home_classification) = 'fbs'
        AND LOWER(away_classification) = 'fbs'

    ORDER BY start_date

    LIMIT 15
    """
).fetchall()


for row in upcoming:

    (
        game_id,
        week,
        start_date,
        away_team,
        home_team,
        away_elo,
        home_elo,
        neutral_site
    ) = row


    print(
        f"{game_id} | "
        f"W{week} | "
        f"{away_team} @ {home_team} | "
        f"away Elo={away_elo} | "
        f"home Elo={home_elo} | "
        f"neutral={neutral_site}"
    )


connection.close()


# ============================================================
# SOURCE AUDIT
#
# We are deliberately inspecting the existing builders and
# trainer instead of assuming how they selected features.
# ============================================================

print_matching_lines(

    TRAIN_SCRIPT,

    [
        "feature",
        "diff_pregame",
        "neutral",
        "elo",
        "talent",
        "return",
        "qb",
        "standardscaler",
        "scaler",
        "input_dim"
    ],

    "TRAIN_NEURAL_NETWORK_V1 RELEVANT LINES"
)


print_matching_lines(

    BUILD_V1_SCRIPT,

    [
        "home_points",
        "away_points",
        "pregame",
        "rolling",
        "shift",
        "game_id",
        "to_csv",
        "dropna"
    ],

    "BUILD_FEATURES_V1 RELEVANT LINES"
)


print_matching_lines(

    QB_SCRIPT,

    [
        "features_v3",
        "features_v4",
        "qb",
        "passing",
        "pregame",
        "shift",
        "game_id"
    ],

    "QB FEATURE BUILDER RELEVANT LINES"
)


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
    "This output tells us:"
)


print(
    "  1. The exact historical pregame feature names."
)


print(
    "  2. Which columns the neural-network script actually used."
)


print(
    "  3. Whether the old feature builder can support "
    "upcoming games or needs a dedicated live builder."
)


print(
    "  4. Which 2026 games are ready to receive live "
    "pregame features."
)