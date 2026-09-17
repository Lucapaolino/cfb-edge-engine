import os
import sqlite3

import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

DATABASE_FILE = "data/cfb.db"

OUTPUT_FILE = (
    "data/processed/"
    "market_integrity_audit_v1.csv"
)

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

def american_to_probability(odds):

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


def favorite_from_spread(spread):

    if pd.isna(spread):
        return np.nan

    spread = float(spread)

    if spread < 0:
        return "HOME"

    if spread > 0:
        return "AWAY"

    return "PICK"


def favorite_from_moneyline(
    home_moneyline,
    away_moneyline
):

    if (
        pd.isna(home_moneyline)
        or
        pd.isna(away_moneyline)
    ):

        return np.nan


    home_probability = (
        american_to_probability(
            home_moneyline
        )
    )


    away_probability = (
        american_to_probability(
            away_moneyline
        )
    )


    if home_probability > away_probability:
        return "HOME"

    if away_probability > home_probability:
        return "AWAY"

    return "PICK"


# ============================================================
# LOAD DATA
# ============================================================

print(
    "MARKET DATA INTEGRITY AUDIT — V1"
)

print(
    "=" * 120
)


connection = sqlite3.connect(
    DATABASE_FILE
)


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


lines = lines[
    lines[
        "provider"
    ].isin(
        PROVIDERS
    )
].copy()


for column in [

    "season",
    "week",
    "spread",
    "spread_open",
    "home_moneyline",
    "away_moneyline"
]:

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


print(
    f"Rows loaded:  {len(lines):,}"
)


print(
    f"Unique games: {lines['game_id'].nunique():,}"
)


# ============================================================
# MONEYLINE INTERNAL CONSISTENCY
# ============================================================

lines[
    "both_moneylines_available"
] = (

    lines[
        "home_moneyline"
    ].notna()

    &

    lines[
        "away_moneyline"
    ].notna()
)


lines[
    "both_negative"
] = (

    (
        lines[
            "home_moneyline"
        ]
        <
        0
    )

    &

    (
        lines[
            "away_moneyline"
        ]
        <
        0
    )
)


lines[
    "both_positive"
] = (

    (
        lines[
            "home_moneyline"
        ]
        >
        0
    )

    &

    (
        lines[
            "away_moneyline"
        ]
        >
        0
    )
)


lines[
    "home_raw_probability"
] = lines[
    "home_moneyline"
].apply(
    american_to_probability
)


lines[
    "away_raw_probability"
] = lines[
    "away_moneyline"
].apply(
    american_to_probability
)


lines[
    "probability_sum"
] = (

    lines[
        "home_raw_probability"
    ]

    +

    lines[
        "away_raw_probability"
    ]
)


# ------------------------------------------------------------
# Typical two-way market:
#
# probability sum > 1 because of vig.
#
# Very high values are suspicious.
# Values below 1 are also unusual if both sides are from
# the same sportsbook snapshot.
# ------------------------------------------------------------

lines[
    "probability_sum_below_1"
] = (

    lines[
        "probability_sum"
    ]
    <
    1.0
)


lines[
    "probability_sum_above_1_15"
] = (

    lines[
        "probability_sum"
    ]
    >
    1.15
)


lines[
    "probability_sum_above_1_25"
] = (

    lines[
        "probability_sum"
    ]
    >
    1.25
)


# ============================================================
# FAVORITE AGREEMENT
# ============================================================

lines[
    "spread_favorite"
] = lines[
    "spread"
].apply(
    favorite_from_spread
)


lines[
    "moneyline_favorite"
] = lines.apply(

    lambda row:

        favorite_from_moneyline(

            row[
                "home_moneyline"
            ],

            row[
                "away_moneyline"
            ]
        ),

    axis=1
)


lines[
    "spread_moneyline_disagree"
] = (

    lines[
        "spread_favorite"
    ].notna()

    &

    lines[
        "moneyline_favorite"
    ].notna()

    &

    (
        lines[
            "spread_favorite"
        ]
        !=
        lines[
            "moneyline_favorite"
        ]
    )

    &

    (
        lines[
            "spread_favorite"
        ]
        !=
        "PICK"
    )

    &

    (
        lines[
            "moneyline_favorite"
        ]
        !=
        "PICK"
    )
)


# ============================================================
# OPENING / LATER SPREAD CHECK
# ============================================================

lines[
    "spread_move"
] = (

    lines[
        "spread"
    ]

    -

    lines[
        "spread_open"
    ]
)


lines[
    "abs_spread_move"
] = lines[
    "spread_move"
].abs()


lines[
    "opening_favorite"
] = lines[
    "spread_open"
].apply(
    favorite_from_spread
)


lines[
    "large_move_5_plus"
] = (

    lines[
        "abs_spread_move"
    ]
    >=
    5.0
)


lines[
    "extreme_move_8_plus"
] = (

    lines[
        "abs_spread_move"
    ]
    >=
    8.0
)


lines[
    "extreme_move_10_plus"
] = (

    lines[
        "abs_spread_move"
    ]
    >=
    10.0
)


# ============================================================
# POSSIBLE OPENING-SIGN REVERSAL
#
# If the opening line becomes MUCH closer to the later spread
# after multiplying opening spread by -1, that suggests a
# possible orientation/sign problem.
#
# Example:
#
# open = -23.5
# later = +13
#
# normal difference:
# |-23.5 - 13| = 36.5
#
# flipped opening:
# |+23.5 - 13| = 10.5
#
# Still not perfect, but much more plausible.
# ============================================================

lines[
    "normal_open_difference"
] = (

    lines[
        "spread_open"
    ]

    -

    lines[
        "spread"
    ]

).abs()


lines[
    "flipped_open_difference"
] = (

    (
        -1.0
        *
        lines[
            "spread_open"
        ]
    )

    -

    lines[
        "spread"
    ]

).abs()


lines[
    "opening_sign_flip_much_better"
] = (

    lines[
        "spread_open"
    ].notna()

    &

    lines[
        "spread"
    ].notna()

    &

    (
        lines[
            "flipped_open_difference"
        ]
        +
        3.0

        <
        lines[
            "normal_open_difference"
        ]
    )
)


# ============================================================
# SUMMARY BY PROVIDER / SEASON
# ============================================================

print()

print(
    "=" * 120
)

print(
    "MONEYLINE INTEGRITY BY PROVIDER / SEASON"
)

print(
    "=" * 120
)


summary_rows = []


for (
    season,
    provider
), subset in lines.groupby(
    [
        "season",
        "provider"
    ]
):

    ml = subset[
        subset[
            "both_moneylines_available"
        ]
    ]


    if len(ml) == 0:

        summary_rows.append(
            {
                "season":
                    int(
                        season
                    ),

                "provider":
                    provider,

                "moneyline_games":
                    0,

                "both_negative_rate":
                    np.nan,

                "both_positive_rate":
                    np.nan,

                "prob_sum_below_1_rate":
                    np.nan,

                "prob_sum_above_1_15_rate":
                    np.nan,

                "prob_sum_above_1_25_rate":
                    np.nan,

                "spread_ml_disagreement":
                    np.nan,

                "mean_probability_sum":
                    np.nan
            }
        )

        continue


    summary_rows.append(
        {
            "season":
                int(
                    season
                ),

            "provider":
                provider,

            "moneyline_games":
                len(
                    ml
                ),

            "both_negative_rate":
                ml[
                    "both_negative"
                ].mean(),

            "both_positive_rate":
                ml[
                    "both_positive"
                ].mean(),

            "prob_sum_below_1_rate":
                ml[
                    "probability_sum_below_1"
                ].mean(),

            "prob_sum_above_1_15_rate":
                ml[
                    "probability_sum_above_1_15"
                ].mean(),

            "prob_sum_above_1_25_rate":
                ml[
                    "probability_sum_above_1_25"
                ].mean(),

            "spread_ml_disagreement":
                ml[
                    "spread_moneyline_disagree"
                ].mean(),

            "mean_probability_sum":
                ml[
                    "probability_sum"
                ].mean()
        }
    )


summary = pd.DataFrame(
    summary_rows
)


print(

    summary.to_string(

        index=False,

        formatters={

            "both_negative_rate":
                lambda x:
                    f"{x:.2%}",

            "both_positive_rate":
                lambda x:
                    f"{x:.2%}",

            "prob_sum_below_1_rate":
                lambda x:
                    f"{x:.2%}",

            "prob_sum_above_1_15_rate":
                lambda x:
                    f"{x:.2%}",

            "prob_sum_above_1_25_rate":
                lambda x:
                    f"{x:.2%}",

            "spread_ml_disagreement":
                lambda x:
                    f"{x:.2%}",

            "mean_probability_sum":
                lambda x:
                    f"{x:.4f}"
        }
    )
)


# ============================================================
# WORST MONEYLINE ROWS
# ============================================================

bad_moneylines = lines[
    lines[
        "both_moneylines_available"
    ]
].copy()


bad_moneylines[
    "suspicion_score"
] = (

    bad_moneylines[
        "both_negative"
    ].astype(
        int
    )
    *
    5

    +

    bad_moneylines[
        "both_positive"
    ].astype(
        int
    )
    *
    2

    +

    bad_moneylines[
        "probability_sum_above_1_25"
    ].astype(
        int
    )
    *
    5

    +

    bad_moneylines[
        "probability_sum_above_1_15"
    ].astype(
        int
    )
    *
    2

    +

    bad_moneylines[
        "spread_moneyline_disagree"
    ].astype(
        int
    )
    *
    2
)


bad_moneylines = (

    bad_moneylines

    .sort_values(
        [
            "suspicion_score",
            "probability_sum"
        ],

        ascending=[
            False,
            False
        ]
    )

    .head(
        40
    )
)


print()

print(
    "=" * 120
)

print(
    "40 MOST SUSPICIOUS MONEYLINE ROWS"
)

print(
    "=" * 120
)


print(

    bad_moneylines[
        [
            "season",
            "week",
            "provider",
            "away_team",
            "home_team",
            "spread",
            "home_moneyline",
            "away_moneyline",
            "probability_sum",
            "both_negative",
            "both_positive",
            "spread_favorite",
            "moneyline_favorite",
            "spread_moneyline_disagree",
            "suspicion_score"
        ]
    ].to_string(

        index=False,

        formatters={

            "spread":
                lambda x:
                    f"{x:+.1f}"
                    if pd.notna(x)
                    else "NaN",

            "home_moneyline":
                lambda x:
                    f"{x:+.0f}",

            "away_moneyline":
                lambda x:
                    f"{x:+.0f}",

            "probability_sum":
                lambda x:
                    f"{x:.4f}"
        }
    )
)


# ============================================================
# SPREAD MOVEMENT INTEGRITY
# ============================================================

movement = lines[
    lines[
        "provider"
    ]
    ==
    "Bovada"
].dropna(
    subset=[
        "spread",
        "spread_open"
    ]
).copy()


print()

print(
    "=" * 120
)

print(
    "BOVADA OPENING-SPREAD INTEGRITY"
)

print(
    "=" * 120
)


movement_rows = []


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


    movement_rows.append(
        {
            "season":
                season,

            "games":
                len(
                    subset
                ),

            "move_5_plus":
                subset[
                    "large_move_5_plus"
                ].mean(),

            "move_8_plus":
                subset[
                    "extreme_move_8_plus"
                ].mean(),

            "move_10_plus":
                subset[
                    "extreme_move_10_plus"
                ].mean(),

            "sign_flip_much_better":
                subset[
                    "opening_sign_flip_much_better"
                ].mean()
        }
    )


movement_summary = pd.DataFrame(
    movement_rows
)


print(

    movement_summary.to_string(

        index=False,

        formatters={

            "move_5_plus":
                lambda x:
                    f"{x:.2%}",

            "move_8_plus":
                lambda x:
                    f"{x:.2%}",

            "move_10_plus":
                lambda x:
                    f"{x:.2%}",

            "sign_flip_much_better":
                lambda x:
                    f"{x:.2%}"
        }
    )
)


# ============================================================
# EXTREME OPENING SPREAD ROWS
# ============================================================

extreme_moves = (

    movement[
        movement[
            "abs_spread_move"
        ]
        >=
        7.0
    ]

    .sort_values(
        "abs_spread_move",
        ascending=False
    )
)


print()

print(
    "=" * 120
)

print(
    "EXTREME BOVADA OPENING/LATER SPREAD ROWS"
)

print(
    "=" * 120
)


print(

    extreme_moves[
        [
            "season",
            "week",
            "away_team",
            "home_team",
            "spread_open",
            "spread",
            "spread_move",
            "normal_open_difference",
            "flipped_open_difference",
            "opening_sign_flip_much_better"
        ]
    ].head(
        50
    ).to_string(

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
                    f"{x:+.1f}",

            "normal_open_difference":
                lambda x:
                    f"{x:.1f}",

            "flipped_open_difference":
                lambda x:
                    f"{x:.1f}"
        }
    )
)


# ============================================================
# COUNTS WE CAN USE FOR CLEANING
# ============================================================

clean_moneyline = (

    lines[
        "both_moneylines_available"
    ]

    &

    ~lines[
        "both_negative"
    ]

    &

    (
        lines[
            "probability_sum"
        ]
        >=
        1.0
    )

    &

    (
        lines[
            "probability_sum"
        ]
        <=
        1.15
    )
)


lines[
    "passes_basic_moneyline_integrity"
] = clean_moneyline


print()

print(
    "=" * 120
)

print(
    "BASIC CLEAN-MONEYLINE RETENTION"
)

print(
    "=" * 120
)


retention = (

    lines[
        lines[
            "both_moneylines_available"
        ]
    ]

    .groupby(
        [
            "season",
            "provider"
        ]
    )

    .agg(

        moneyline_games=(
            "game_id",
            "size"
        ),

        clean_games=(
            "passes_basic_moneyline_integrity",
            "sum"
        )
    )

    .reset_index()
)


retention[
    "retention"
] = (

    retention[
        "clean_games"
    ]

    /

    retention[
        "moneyline_games"
    ]
)


print(

    retention.to_string(

        index=False,

        formatters={

            "retention":
                lambda x:
                    f"{x:.2%}"
        }
    )
)


# ============================================================
# SAVE FULL AUDIT
# ============================================================

os.makedirs(
    "data/processed",
    exist_ok=True
)


lines.to_csv(
    OUTPUT_FILE,
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
    OUTPUT_FILE
)


print()

print(
    "No betting rows were deleted or modified."
)