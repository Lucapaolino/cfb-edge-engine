import os
import re
import sqlite3

import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

DATABASE_FILE = (
    "data/cfb.db"
)

FEATURE_FILE = (
    "data/processed/"
    "features_v2.csv"
)

COVERAGE_FILE = (
    "data/processed/"
    "moneyline_coverage_by_season.csv"
)

PROVIDER_FILE = (
    "data/processed/"
    "moneyline_coverage_by_provider.csv"
)


# ============================================================
# HELPERS
# ============================================================

def normalize_column_name(
    column
):

    return re.sub(
        r"[^a-z0-9]",
        "",
        column.lower()
    )


def quote_identifier(
    column
):

    return (
        '"'
        +
        column.replace(
            '"',
            '""'
        )
        +
        '"'
    )


# ============================================================
# OPEN DATABASE
# ============================================================

print(
    "MONEYLINE DATA AUDIT"
)

print(
    "=" * 100
)


if not os.path.exists(
    DATABASE_FILE
):

    raise FileNotFoundError(
        DATABASE_FILE
    )


connection = sqlite3.connect(
    DATABASE_FILE
)


# ============================================================
# INSPECT BETTING LINE TABLE
# ============================================================

table_info = pd.read_sql_query(

    "PRAGMA table_info(betting_lines)",

    connection
)


if len(
    table_info
) == 0:

    raise RuntimeError(
        "betting_lines table does not exist "
        "or contains no schema information."
    )


columns = table_info[
    "name"
].tolist()


print()

print(
    "BETTING_LINES COLUMNS"
)

print(
    "-" * 100
)


for column in columns:

    print(
        column
    )


# ============================================================
# DETECT IMPORTANT COLUMNS
# ============================================================

normalized = {

    column:
        normalize_column_name(
            column
        )

    for column in columns
}


# ------------------------------------------------------------
# Game ID
# ------------------------------------------------------------

game_id_column = None


for (
    column,
    normalized_name
) in normalized.items():

    if normalized_name == "gameid":

        game_id_column = column

        break


# ------------------------------------------------------------
# Home moneyline
# ------------------------------------------------------------

home_moneyline_column = None


for (
    column,
    normalized_name
) in normalized.items():

    if (

        "home"
        in normalized_name

        and

        "moneyline"
        in normalized_name
    ):

        home_moneyline_column = column

        break


# ------------------------------------------------------------
# Away moneyline
# ------------------------------------------------------------

away_moneyline_column = None


for (
    column,
    normalized_name
) in normalized.items():

    if (

        "away"
        in normalized_name

        and

        "moneyline"
        in normalized_name
    ):

        away_moneyline_column = column

        break


# ------------------------------------------------------------
# Provider
# ------------------------------------------------------------

provider_column = None


for (
    column,
    normalized_name
) in normalized.items():

    if "provider" in normalized_name:

        provider_column = column

        break


# ============================================================
# REPORT DETECTION
# ============================================================

print()

print(
    "DETECTED COLUMNS"
)

print(
    "-" * 100
)


print(
    f"Game ID:        "
    f"{game_id_column}"
)

print(
    f"Home moneyline: "
    f"{home_moneyline_column}"
)

print(
    f"Away moneyline: "
    f"{away_moneyline_column}"
)

print(
    f"Provider:       "
    f"{provider_column}"
)


# ============================================================
# SAFETY CHECKS
# ============================================================

if game_id_column is None:

    raise RuntimeError(
        "Could not identify the game ID column."
    )


if home_moneyline_column is None:

    raise RuntimeError(
        "Could not identify the home moneyline column."
    )


if away_moneyline_column is None:

    raise RuntimeError(
        "Could not identify the away moneyline column."
    )


# ============================================================
# LOAD MONEYLINES
# ============================================================

select_parts = [

    (
        f"{quote_identifier(game_id_column)} "
        f"AS game_id"
    ),

    (
        f"{quote_identifier(home_moneyline_column)} "
        f"AS home_moneyline"
    ),

    (
        f"{quote_identifier(away_moneyline_column)} "
        f"AS away_moneyline"
    )
]


if provider_column is not None:

    select_parts.append(

        f"{quote_identifier(provider_column)} "
        f"AS provider"
    )


query = (

    "SELECT "

    +

    ", ".join(
        select_parts
    )

    +

    " FROM betting_lines"
)


moneylines = pd.read_sql_query(

    query,

    connection
)


connection.close()


if provider_column is None:

    moneylines[
        "provider"
    ] = "Unknown"


print()

print(
    f"Betting-line rows loaded: "
    f"{len(moneylines):,}"
)


# ============================================================
# CLEAN VALUES
# ============================================================

moneylines[
    "game_id"
] = pd.to_numeric(

    moneylines[
        "game_id"
    ],

    errors="coerce"
).astype(
    "Int64"
)


moneylines[
    "home_moneyline"
] = pd.to_numeric(

    moneylines[
        "home_moneyline"
    ],

    errors="coerce"
)


moneylines[
    "away_moneyline"
] = pd.to_numeric(

    moneylines[
        "away_moneyline"
    ],

    errors="coerce"
)


# ------------------------------------------------------------
# American moneyline 0 is not a valid betting price.
# ------------------------------------------------------------

moneylines.loc[
    moneylines[
        "home_moneyline"
    ] == 0,
    "home_moneyline"
] = np.nan


moneylines.loc[
    moneylines[
        "away_moneyline"
    ] == 0,
    "away_moneyline"
] = np.nan


moneylines[
    "provider"
] = (

    moneylines[
        "provider"
    ]

    .fillna(
        "Unknown"
    )

    .astype(
        str
    )
)


# ============================================================
# LOAD EXACT MODEL POPULATION
# ============================================================

model_games = pd.read_csv(

    FEATURE_FILE,

    usecols=[
        "game_id",
        "season"
    ]
)


model_games[
    "game_id"
] = pd.to_numeric(

    model_games[
        "game_id"
    ],

    errors="coerce"
).astype(
    "Int64"
)


model_games[
    "season"
] = pd.to_numeric(

    model_games[
        "season"
    ],

    errors="coerce"
).astype(
    "Int64"
)


model_games = (

    model_games

    .dropna(
        subset=[
            "game_id",
            "season"
        ]
    )

    .drop_duplicates(
        subset=[
            "game_id"
        ]
    )
)


print(
    f"Model games loaded:        "
    f"{len(model_games):,}"
)


# ============================================================
# VALID PAIRED MONEYLINES
#
# We need BOTH sides.
#
# Without both sides, we cannot properly remove the vig.
# ============================================================

moneylines[
    "paired_moneyline"
] = (

    moneylines[
        "home_moneyline"
    ].notna()

    &

    moneylines[
        "away_moneyline"
    ].notna()
)


paired_lines = (

    moneylines[
        moneylines[
            "paired_moneyline"
        ]
    ]

    .copy()
)


# ------------------------------------------------------------
# Only games actually belonging to our model population.
# ------------------------------------------------------------

paired_lines = paired_lines.merge(

    model_games,

    on="game_id",

    how="inner"
)


print(
    f"Rows with both moneylines: "
    f"{len(paired_lines):,}"
)


print(
    f"Unique model games with "
    f"paired moneylines: "
    f"{paired_lines['game_id'].nunique():,}"
)


# ============================================================
# COVERAGE BY SEASON
# ============================================================

coverage_rows = []


for season in sorted(
    model_games[
        "season"
    ].dropna().unique()
):

    season_games = model_games[
        model_games[
            "season"
        ]
        ==
        season
    ]


    season_paired = paired_lines[
        paired_lines[
            "season"
        ]
        ==
        season
    ]


    total_games = (
        season_games[
            "game_id"
        ]
        .nunique()
    )


    covered_games = (
        season_paired[
            "game_id"
        ]
        .nunique()
    )


    coverage_pct = (

        covered_games

        /
        total_games

        *
        100
    )


    coverage_rows.append(
        {
            "season":
                int(
                    season
                ),

            "model_games":
                total_games,

            "games_with_paired_moneylines":
                covered_games,

            "coverage_pct":
                coverage_pct
        }
    )


coverage = pd.DataFrame(
    coverage_rows
)


print()

print(
    "PAIRED MONEYLINE COVERAGE BY SEASON"
)

print(
    "-" * 100
)


print(

    coverage.to_string(

        index=False,

        formatters={

            "coverage_pct":
                lambda value:
                    f"{value:.2f}%"
        }
    )
)


# ============================================================
# PROVIDER COVERAGE
# ============================================================

provider_coverage = (

    paired_lines

    .groupby(
        "provider",
        as_index=False
    )

    .agg(

        line_rows=(
            "game_id",
            "size"
        ),

        unique_games=(
            "game_id",
            "nunique"
        ),

        first_season=(
            "season",
            "min"
        ),

        last_season=(
            "season",
            "max"
        )
    )

    .sort_values(
        [
            "unique_games",
            "line_rows"
        ],

        ascending=False
    )
)


print()

print(
    "TOP PROVIDERS WITH PAIRED MONEYLINES"
)

print(
    "-" * 100
)


print(

    provider_coverage

    .head(
        20
    )

    .to_string(
        index=False
    )
)


# ============================================================
# SAMPLE VALID MONEYLINES
# ============================================================

print()

print(
    "SAMPLE VALID MONEYLINES"
)

print(
    "-" * 100
)


sample_columns = [

    "season",

    "game_id",

    "provider",

    "home_moneyline",

    "away_moneyline"
]


print(

    paired_lines[
        sample_columns
    ]

    .head(
        20
    )

    .to_string(
        index=False
    )
)


# ============================================================
# SAVE AUDIT FILES
# ============================================================

os.makedirs(

    "data/processed",

    exist_ok=True
)


coverage.to_csv(

    COVERAGE_FILE,

    index=False
)


provider_coverage.to_csv(

    PROVIDER_FILE,

    index=False
)


# ============================================================
# FINISH
# ============================================================

print()

print(
    "=" * 100
)

print(
    "AUDIT COMPLETE"
)

print(
    "=" * 100
)


print(
    COVERAGE_FILE
)

print(
    PROVIDER_FILE
)