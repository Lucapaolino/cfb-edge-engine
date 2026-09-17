import os
import sqlite3

import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

DATABASE_FILE = "data/cfb.db"

OUTPUT_DIRECTORY = "data/processed"

PROVIDERS = [
    "Bovada",
    "DraftKings",
    "ESPN Bet"
]

SEASONS = [
    2021,
    2022,
    2023,
    2024,
    2025
]


# ============================================================
# HELPERS
# ============================================================

def american_to_implied_probability(odds):

    if pd.isna(odds):
        return np.nan

    odds = float(odds)

    if odds < 0:

        return (
            abs(odds)
            /
            (
                abs(odds)
                +
                100.0
            )
        )

    if odds > 0:

        return (
            100.0
            /
            (
                odds
                +
                100.0
            )
        )

    return np.nan


def american_to_decimal_odds(odds):

    if pd.isna(odds):
        return np.nan

    odds = float(odds)

    if odds < 0:

        return (
            1.0
            +
            (
                100.0
                /
                abs(odds)
            )
        )

    if odds > 0:

        return (
            1.0
            +
            (
                odds
                /
                100.0
            )
        )

    return np.nan


# ============================================================
# START
# ============================================================

print(
    "MARKET MOVEMENT + CROSS-BOOK AUDIT — V1"
)

print(
    "=" * 120
)


connection = sqlite3.connect(
    DATABASE_FILE
)


# ============================================================
# LOAD BETTING LINES
# ============================================================

lines = pd.read_sql_query(

    """
    SELECT
        b.game_id,
        g.season,
        g.week,
        g.home_team,
        g.away_team,
        b.provider,
        b.spread,
        b.spread_open,
        b.over_under,
        b.over_under_open,
        b.home_moneyline,
        b.away_moneyline

    FROM betting_lines b

    JOIN games g
        ON g.game_id = b.game_id

    WHERE g.season BETWEEN 2021 AND 2025
    """,

    connection
)


connection.close()


print(
    f"Betting rows loaded: "
    f"{len(lines):,}"
)


print(
    f"Unique games:        "
    f"{lines['game_id'].nunique():,}"
)


# ============================================================
# CLEAN NUMERIC COLUMNS
# ============================================================

numeric_columns = [

    "season",
    "week",
    "spread",
    "spread_open",
    "over_under",
    "over_under_open",
    "home_moneyline",
    "away_moneyline"
]


for column in numeric_columns:

    lines[
        column
    ] = pd.to_numeric(

        lines[
            column
        ],

        errors="coerce"
    )


for column in [

    "home_moneyline",
    "away_moneyline"
]:

    lines.loc[
        lines[
            column
        ]
        ==
        0,
        column
    ] = np.nan


# ============================================================
# BASIC PROVIDER COVERAGE
# ============================================================

print()

print(
    "=" * 120
)

print(
    "PROVIDER COVERAGE"
)

print(
    "=" * 120
)


provider_coverage = (

    lines[
        lines[
            "provider"
        ].isin(
            PROVIDERS
        )
    ]

    .groupby(
        [
            "season",
            "provider"
        ]
    )

    .agg(

        rows=(
            "game_id",
            "size"
        ),

        unique_games=(
            "game_id",
            "nunique"
        ),

        spread_games=(
            "spread",
            lambda x:
                x.notna().sum()
        ),

        opening_spread_games=(
            "spread_open",
            lambda x:
                x.notna().sum()
        ),

        moneyline_games=(
            "home_moneyline",
            lambda x:
                x.notna().sum()
        )
    )

    .reset_index()
)


print(

    provider_coverage.to_string(
        index=False
    )
)


# ============================================================
# BOVADA LINE MOVEMENT
#
# spread_open = earlier/opening stored spread
# spread      = later/latest stored spread
#
# IMPORTANT:
# We are NOT calling spread a verified closing line.
# ============================================================

bovada = lines[
    lines[
        "provider"
    ]
    ==
    "Bovada"
].copy()


bovada[
    "spread_move"
] = (

    bovada[
        "spread"
    ]

    -

    bovada[
        "spread_open"
    ]
)


bovada[
    "abs_spread_move"
] = bovada[
    "spread_move"
].abs()


# ============================================================
# FAVORITE SIDE
#
# Negative home spread = home favorite
# Positive home spread = away favorite
# Zero = pick'em
# ============================================================

def favorite_side(spread):

    if pd.isna(spread):
        return np.nan

    if spread < 0:
        return "HOME"

    if spread > 0:
        return "AWAY"

    return "PICK"


bovada[
    "opening_favorite"
] = bovada[
    "spread_open"
].apply(
    favorite_side
)


bovada[
    "later_favorite"
] = bovada[
    "spread"
].apply(
        favorite_side
)


bovada[
    "favorite_flip"
] = (

    bovada[
        "opening_favorite"
    ].notna()

    &

    bovada[
        "later_favorite"
    ].notna()

    &

    (
        bovada[
            "opening_favorite"
        ]

        !=

        bovada[
            "later_favorite"
        ]
    )

    &

    (
        bovada[
            "opening_favorite"
        ]
        !=
        "PICK"
    )

    &

    (
        bovada[
            "later_favorite"
        ]
        !=
        "PICK"
    )
)


# ============================================================
# OPENING SPREAD COVERAGE
# ============================================================

print()

print(
    "=" * 120
)

print(
    "BOVADA OPENING-SPREAD COVERAGE"
)

print(
    "=" * 120
)


opening_coverage = (

    bovada

    .groupby(
        "season"
    )

    .agg(

        games=(
            "game_id",
            "nunique"
        ),

        opening_spread_available=(
            "spread_open",
            lambda x:
                x.notna().sum()
        )
    )

    .reset_index()
)


opening_coverage[
    "coverage"
] = (

    opening_coverage[
        "opening_spread_available"
    ]

    /

    opening_coverage[
        "games"
    ]
)


print(

    opening_coverage.to_string(

        index=False,

        formatters={

            "coverage":
                lambda x:
                    f"{x:.2%}"
        }
    )
)


# ============================================================
# LINE MOVEMENT DISTRIBUTION
# ============================================================

movement = bovada.dropna(
    subset=[
        "spread_open",
        "spread"
    ]
).copy()


print()

print(
    "=" * 120
)

print(
    "BOVADA SPREAD MOVEMENT BY SEASON"
)

print(
    "=" * 120
)


movement_summary_rows = []


for season in SEASONS:

    subset = movement[
        movement[
            "season"
        ]
        ==
        season
    ]


    if len(subset) == 0:
        continue


    movement_summary_rows.append(
        {
            "season":
                season,

            "games":
                len(
                    subset
                ),

            "mean_signed_move":
                subset[
                    "spread_move"
                ].mean(),

            "mean_absolute_move":
                subset[
                    "abs_spread_move"
                ].mean(),

            "median_absolute_move":
                subset[
                    "abs_spread_move"
                ].median(),

            "pct_move_1_plus":
                (
                    subset[
                        "abs_spread_move"
                    ]
                    >=
                    1.0
                ).mean(),

            "pct_move_2_plus":
                (
                    subset[
                        "abs_spread_move"
                    ]
                    >=
                    2.0
                ).mean(),

            "pct_move_3_plus":
                (
                    subset[
                        "abs_spread_move"
                    ]
                    >=
                    3.0
                ).mean(),

            "favorite_flip_rate":
                subset[
                    "favorite_flip"
                ].mean()
        }
    )


movement_summary = pd.DataFrame(
    movement_summary_rows
)


print(

    movement_summary.to_string(

        index=False,

        formatters={

            "mean_signed_move":
                lambda x:
                    f"{x:+.3f}",

            "mean_absolute_move":
                lambda x:
                    f"{x:.3f}",

            "median_absolute_move":
                lambda x:
                    f"{x:.3f}",

            "pct_move_1_plus":
                lambda x:
                    f"{x:.2%}",

            "pct_move_2_plus":
                lambda x:
                    f"{x:.2%}",

            "pct_move_3_plus":
                lambda x:
                    f"{x:.2%}",

            "favorite_flip_rate":
                lambda x:
                    f"{x:.2%}"
        }
    )
)


# ============================================================
# LINE MOVEMENT BUCKETS
# ============================================================

def movement_bucket(value):

    if pd.isna(value):
        return "UNKNOWN"

    value = abs(
        float(
            value
        )
    )

    if value < 0.5:
        return "<0.5"

    if value < 1.0:
        return "0.5-0.99"

    if value < 2.0:
        return "1.0-1.99"

    if value < 3.0:
        return "2.0-2.99"

    if value < 5.0:
        return "3.0-4.99"

    return "5.0+"


movement[
    "movement_bucket"
] = movement[
    "abs_spread_move"
].apply(
    movement_bucket
)


movement_bucket_summary = (

    movement

    .groupby(
        "movement_bucket"
    )

    .agg(

        games=(
            "game_id",
            "size"
        ),

        mean_abs_move=(
            "abs_spread_move",
            "mean"
        )
    )

    .reset_index()
)


print()

print(
    "=" * 120
)

print(
    "BOVADA SPREAD-MOVEMENT BUCKETS"
)

print(
    "=" * 120
)


print(

    movement_bucket_summary.to_string(

        index=False,

        formatters={

            "mean_abs_move":
                lambda x:
                    f"{x:.3f}"
        }
    )
)


# ============================================================
# LARGEST MOVES
# ============================================================

largest_moves = (

    movement

    .sort_values(
        "abs_spread_move",
        ascending=False
    )

    .head(
        25
    )
)


print()

print(
    "=" * 120
)

print(
    "25 LARGEST BOVADA SPREAD MOVES"
)

print(
    "=" * 120
)


print(

    largest_moves[
        [
            "season",
            "week",
            "away_team",
            "home_team",
            "spread_open",
            "spread",
            "spread_move",
            "opening_favorite",
            "later_favorite"
        ]
    ].to_string(

        index=False,

        formatters={

            "spread_open":
                lambda x:
                    f"{x:+.1f}",

            "spread":
                lambda x:
                    f"{x:+.1f}",

            "spread_move":
                lambda x:
                    f"{x:+.1f}"
        }
    )
)


# ============================================================
# CROSS-BOOK MONEYLINE DATA
# ============================================================

moneylines = lines[
    lines[
        "provider"
    ].isin(
        PROVIDERS
    )
].copy()


moneylines = moneylines.dropna(
    subset=[
        "home_moneyline",
        "away_moneyline"
    ]
)


# ============================================================
# CONVERT TO DECIMAL
# ============================================================

moneylines[
    "home_decimal"
] = moneylines[
    "home_moneyline"
].apply(
    american_to_decimal_odds
)


moneylines[
    "away_decimal"
] = moneylines[
    "away_moneyline"
].apply(
    american_to_decimal_odds
)


moneylines[
    "home_raw_probability"
] = moneylines[
    "home_moneyline"
].apply(
    american_to_implied_probability
)


moneylines[
    "away_raw_probability"
] = moneylines[
    "away_moneyline"
].apply(
    american_to_implied_probability
)


# ============================================================
# CROSS-BOOK GAME SUMMARY
#
# Decimal odds:
# higher = better payout for bettor.
# ============================================================

cross_book_rows = []


for game_id, group in moneylines.groupby(
    "game_id"
):

    if group[
        "provider"
    ].nunique() < 2:

        continue


    first = group.iloc[
        0
    ]


    best_home_row = group.loc[
        group[
            "home_decimal"
        ].idxmax()
    ]


    worst_home_row = group.loc[
        group[
            "home_decimal"
        ].idxmin()
    ]


    best_away_row = group.loc[
        group[
            "away_decimal"
        ].idxmax()
    ]


    worst_away_row = group.loc[
        group[
            "away_decimal"
        ].idxmin()
    ]


    home_decimal_gap = (

        float(
            best_home_row[
                "home_decimal"
            ]
        )

        -

        float(
            worst_home_row[
                "home_decimal"
            ]
        )
    )


    away_decimal_gap = (

        float(
            best_away_row[
                "away_decimal"
            ]
        )

        -

        float(
            worst_away_row[
                "away_decimal"
            ]
        )
    )


    home_probability_range = (

        group[
            "home_raw_probability"
        ].max()

        -

        group[
            "home_raw_probability"
        ].min()
    )


    away_probability_range = (

        group[
            "away_raw_probability"
        ].max()

        -

        group[
            "away_raw_probability"
        ].min()
    )


    cross_book_rows.append(
        {
            "game_id":
                game_id,

            "season":
                int(
                    first[
                        "season"
                    ]
                ),

            "week":
                int(
                    first[
                        "week"
                    ]
                ),

            "home_team":
                first[
                    "home_team"
                ],

            "away_team":
                first[
                    "away_team"
                ],

            "books":
                group[
                    "provider"
                ].nunique(),

            "home_best_provider":
                best_home_row[
                    "provider"
                ],

            "home_best_moneyline":
                best_home_row[
                    "home_moneyline"
                ],

            "home_worst_provider":
                worst_home_row[
                    "provider"
                ],

            "home_worst_moneyline":
                worst_home_row[
                    "home_moneyline"
                ],

            "home_decimal_gap":
                home_decimal_gap,

            "home_probability_range":
                home_probability_range,

            "away_best_provider":
                best_away_row[
                    "provider"
                ],

            "away_best_moneyline":
                best_away_row[
                    "away_moneyline"
                ],

            "away_worst_provider":
                worst_away_row[
                    "provider"
                ],

            "away_worst_moneyline":
                worst_away_row[
                    "away_moneyline"
                ],

            "away_decimal_gap":
                away_decimal_gap,

            "away_probability_range":
                away_probability_range
        }
    )


cross_book = pd.DataFrame(
    cross_book_rows
)


print()

print(
    "=" * 120
)

print(
    "CROSS-BOOK MONEYLINE COVERAGE"
)

print(
    "=" * 120
)


if len(
    cross_book
) == 0:

    print(
        "No games contained two or more selected sportsbooks."
    )


else:

    cross_book_coverage = (

        cross_book

        .groupby(
            "season"
        )

        .agg(

            games_with_2plus_books=(
                "game_id",
                "size"
            ),

            average_books=(
                "books",
                "mean"
            ),

            mean_home_probability_range=(
                "home_probability_range",
                "mean"
            ),

            mean_away_probability_range=(
                "away_probability_range",
                "mean"
            ),

            pct_home_disagreement_2pp_plus=(
                "home_probability_range",
                lambda x:
                    (
                        x
                        >=
                        0.02
                    ).mean()
            ),

            pct_away_disagreement_2pp_plus=(
                "away_probability_range",
                lambda x:
                    (
                        x
                        >=
                        0.02
                    ).mean()
            )
        )

        .reset_index()
    )


    print(

        cross_book_coverage.to_string(

            index=False,

            formatters={

                "average_books":
                    lambda x:
                        f"{x:.2f}",

                "mean_home_probability_range":
                    lambda x:
                        f"{x:.2%}",

                "mean_away_probability_range":
                    lambda x:
                        f"{x:.2%}",

                "pct_home_disagreement_2pp_plus":
                    lambda x:
                        f"{x:.2%}",

                "pct_away_disagreement_2pp_plus":
                    lambda x:
                        f"{x:.2%}"
            }
        )
    )


# ============================================================
# LARGEST CROSS-BOOK DISAGREEMENTS
# ============================================================

if len(
    cross_book
) > 0:

    cross_book[
        "max_probability_range"
    ] = cross_book[
        [
            "home_probability_range",
            "away_probability_range"
        ]
    ].max(
        axis=1
    )


    biggest_disagreement = (

        cross_book

        .sort_values(
            "max_probability_range",
            ascending=False
        )

        .head(
            25
        )
    )


    print()

    print(
        "=" * 120
    )

    print(
        "25 LARGEST CROSS-BOOK MONEYLINE DISAGREEMENTS"
    )

    print(
        "=" * 120
    )


    print(

        biggest_disagreement[
            [
                "season",
                "week",
                "away_team",
                "home_team",
                "books",
                "home_best_provider",
                "home_best_moneyline",
                "home_worst_provider",
                "home_worst_moneyline",
                "home_probability_range",
                "away_best_provider",
                "away_best_moneyline",
                "away_worst_provider",
                "away_worst_moneyline",
                "away_probability_range"
            ]
        ].to_string(

            index=False,

            formatters={

                "home_best_moneyline":
                    lambda x:
                        f"{x:+.0f}",

                "home_worst_moneyline":
                    lambda x:
                        f"{x:+.0f}",

                "home_probability_range":
                    lambda x:
                        f"{x:.2%}",

                "away_best_moneyline":
                    lambda x:
                        f"{x:+.0f}",

                "away_worst_moneyline":
                    lambda x:
                        f"{x:+.0f}",

                "away_probability_range":
                    lambda x:
                        f"{x:.2%}"
            }
        )
    )


# ============================================================
# PROVIDER PAIRS
# ============================================================

print()

print(
    "=" * 120
)

print(
    "PROVIDER PAIR OVERLAP"
)

print(
    "=" * 120
)


pair_rows = []


for index, provider_a in enumerate(
    PROVIDERS
):

    for provider_b in PROVIDERS[
        index + 1:
    ]:

        a_games = set(

            moneylines.loc[
                moneylines[
                    "provider"
                ]
                ==
                provider_a,
                "game_id"
            ]
        )


        b_games = set(

            moneylines.loc[
                moneylines[
                    "provider"
                ]
                ==
                provider_b,
                "game_id"
            ]
        )


        overlap = a_games.intersection(
            b_games
        )


        pair_rows.append(
            {
                "provider_a":
                    provider_a,

                "provider_b":
                    provider_b,

                "overlap_games":
                    len(
                        overlap
                    )
            }
        )


pair_overlap = pd.DataFrame(
    pair_rows
)


print(
    pair_overlap.to_string(
        index=False
    )
)


# ============================================================
# SAVE OUTPUTS
# ============================================================

os.makedirs(
    OUTPUT_DIRECTORY,
    exist_ok=True
)


provider_coverage.to_csv(

    os.path.join(
        OUTPUT_DIRECTORY,
        "market_provider_coverage_v1.csv"
    ),

    index=False
)


opening_coverage.to_csv(

    os.path.join(
        OUTPUT_DIRECTORY,
        "market_opening_spread_coverage_v1.csv"
    ),

    index=False
)


movement.to_csv(

    os.path.join(
        OUTPUT_DIRECTORY,
        "market_spread_movement_games_v1.csv"
    ),

    index=False
)


movement_summary.to_csv(

    os.path.join(
        OUTPUT_DIRECTORY,
        "market_spread_movement_summary_v1.csv"
    ),

    index=False
)


cross_book.to_csv(

    os.path.join(
        OUTPUT_DIRECTORY,
        "market_cross_book_moneylines_v1.csv"
    ),

    index=False
)


print()

print(
    "=" * 120
)

print(
    "AUDIT COMPLETE"
)

print(
    "=" * 120
)


print(
    "IMPORTANT:"
)


print(
    "Historical line timestamps have NOT been verified."
)


print(
    "spread_open is treated as the stored opening line."
)


print(
    "spread is treated only as the later/latest stored historical line,"
)


print(
    "not automatically as a verified closing line."
)