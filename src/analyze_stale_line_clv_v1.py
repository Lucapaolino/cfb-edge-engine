import os

import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = (
    "data/processed/"
    "market_lead_lag_transitions_v1.csv"
)

OUTPUT_FILE = (
    "data/processed/"
    "stale_line_clv_signals_v1.csv"
)


# ============================================================
# THRESHOLDS
# ============================================================

SPREAD_THRESHOLDS = [
    0.5,
    1.0,
    1.5,
    2.0
]


MONEYLINE_THRESHOLDS_PP = [
    0.5,
    1.0,
    1.5,
    2.0
]


# ============================================================
# START
# ============================================================

print(
    "EVENT-LEVEL STALE LINE + FORWARD CLV TEST — V1"
)

print(
    "=" * 115
)


data = pd.read_csv(
    INPUT_FILE
)


if len(data) == 0:

    raise RuntimeError(
        "Lead/lag transition file is empty."
    )


# ============================================================
# NUMERIC CLEANING
# ============================================================

numeric_columns = [

    "hours_between",
    "hours_until_kickoff",

    "moneyline_consensus_books",
    "moneyline_target_before",
    "moneyline_consensus_before",
    "moneyline_target_after",
    "moneyline_consensus_after",
    "moneyline_initial_deviation",
    "moneyline_initial_abs_deviation",
    "moneyline_final_abs_deviation",
    "moneyline_gap_closed",

    "spread_consensus_books",
    "spread_target_before",
    "spread_consensus_before",
    "spread_target_after",
    "spread_consensus_after",
    "spread_initial_deviation",
    "spread_initial_abs_deviation",
    "spread_final_abs_deviation",
    "spread_gap_closed"
]


for column in numeric_columns:

    if column in data.columns:

        data[
            column
        ] = pd.to_numeric(

            data[
                column
            ],

            errors="coerce"
        )


print(
    f"Raw bookmaker transitions: "
    f"{len(data):,}"
)


print(
    f"Unique games:               "
    f"{data['odds_api_event_id'].nunique():,}"
)


# ============================================================
# IMPORTANT:
#
# ONE TARGET BOOK PER GAME / TRANSITION WINDOW
#
# Suppose eight books exist for the same game.
#
# We DO NOT count all eight as eight independent signals.
#
# For each game and snapshot interval, we select the book
# with the largest disagreement from consensus.
# ============================================================


# ============================================================
# SPREAD EVENT SIGNALS
# ============================================================

spread_candidates = data[

    data[
        "spread_initial_abs_deviation"
    ].notna()

    &

    data[
        "spread_consensus_after"
    ].notna()

].copy()


spread_candidates = spread_candidates.sort_values(

    [
        "odds_api_event_id",
        "current_time",
        "next_time",
        "spread_initial_abs_deviation"
    ],

    ascending=[
        True,
        True,
        True,
        False
    ]
)


spread_event_signals = (

    spread_candidates

    .drop_duplicates(

        subset=[
            "odds_api_event_id",
            "current_time",
            "next_time"
        ],

        keep="first"
    )

    .copy()
)


# ============================================================
# WHAT SIDE GOT THE FAVORABLE SPREAD?
#
# home spread convention:
#
# target = +3.5
# consensus = +2.5
#
# home bettor gets an extra point at target.
# BET SIDE = HOME.
#
#
# target = -4.5
# consensus = -3.5
#
# Target is worse for home but better for the AWAY side:
#
# away target = +4.5
# away consensus = +3.5
#
# BET SIDE = AWAY.
# ============================================================

spread_event_signals[
    "stale_side"
] = np.where(

    spread_event_signals[
        "spread_initial_deviation"
    ]
    >
    0,

    "HOME",

    "AWAY"
)


# ============================================================
# FORWARD SPREAD CLV
#
# Compare the number available at the outlier sportsbook
# NOW against the market consensus at the NEXT snapshot.
#
#
# HOME example:
#
# target now     +3.5
# next consensus +2.5
#
# CLV = +1.0
#
#
# AWAY example:
#
# target home line     -4.5
# next consensus home  -3.5
#
# target away line     +4.5
# next market away     +3.5
#
# CLV = +1.0
#
#
# Positive = stale book gave us a better number.
# Negative = the broader market moved the other way.
# ============================================================

spread_event_signals[
    "forward_clv_points"
] = np.where(

    spread_event_signals[
        "stale_side"
    ]
    ==
    "HOME",

    (
        spread_event_signals[
            "spread_target_before"
        ]

        -

        spread_event_signals[
            "spread_consensus_after"
        ]
    ),

    (
        spread_event_signals[
            "spread_consensus_after"
        ]

        -

        spread_event_signals[
            "spread_target_before"
        ]
    )
)


spread_event_signals[
    "positive_forward_clv"
] = (

    spread_event_signals[
        "forward_clv_points"
    ]
    >
    0
).astype(
    int
)


spread_event_signals[
    "nonnegative_forward_clv"
] = (

    spread_event_signals[
        "forward_clv_points"
    ]
    >=
    0
).astype(
    int
)


# ============================================================
# MONEYLINE EVENT SIGNALS
# ============================================================

moneyline_candidates = data[

    data[
        "moneyline_initial_abs_deviation"
    ].notna()

    &

    data[
        "moneyline_consensus_after"
    ].notna()

].copy()


moneyline_candidates = moneyline_candidates.sort_values(

    [
        "odds_api_event_id",
        "current_time",
        "next_time",
        "moneyline_initial_abs_deviation"
    ],

    ascending=[
        True,
        True,
        True,
        False
    ]
)


moneyline_event_signals = (

    moneyline_candidates

    .drop_duplicates(

        subset=[
            "odds_api_event_id",
            "current_time",
            "next_time"
        ],

        keep="first"
    )

    .copy()
)


# ============================================================
# MONEYLINE SIDE
#
# fair_home_probability LOWER than consensus:
#
# sportsbook is offering HOME more cheaply than market.
#
# fair_home_probability HIGHER than consensus:
#
# sportsbook is offering AWAY more cheaply than market.
# ============================================================

moneyline_event_signals[
    "stale_side"
] = np.where(

    moneyline_event_signals[
        "moneyline_initial_deviation"
    ]
    <
    0,

    "HOME",

    "AWAY"
)


# ============================================================
# FORWARD MONEYLINE CLV
#
# This is measured in FAIR PROBABILITY percentage points,
# not American odds.
#
#
# HOME stale:
#
# target home probability = 45%
# next consensus          = 48%
#
# We obtained the home side when market implied it was less
# likely / cheaper.
#
# forward CLV = +3 pp
#
#
# AWAY stale:
#
# target home probability = 60%
# next consensus          = 57%
#
# Target's away probability was 40%.
# Next market away probability is 43%.
#
# forward CLV = +3 pp.
# ============================================================

moneyline_event_signals[
    "forward_clv_probability"
] = np.where(

    moneyline_event_signals[
        "stale_side"
    ]
    ==
    "HOME",

    (
        moneyline_event_signals[
            "moneyline_consensus_after"
        ]

        -

        moneyline_event_signals[
            "moneyline_target_before"
        ]
    ),

    (
        moneyline_event_signals[
            "moneyline_target_before"
        ]

        -

        moneyline_event_signals[
            "moneyline_consensus_after"
        ]
    )
)


moneyline_event_signals[
    "forward_clv_pp"
] = (

    moneyline_event_signals[
        "forward_clv_probability"
    ]

    *
    100.0
)


moneyline_event_signals[
    "positive_forward_clv"
] = (

    moneyline_event_signals[
        "forward_clv_probability"
    ]
    >
    0
).astype(
    int
)


moneyline_event_signals[
    "nonnegative_forward_clv"
] = (

    moneyline_event_signals[
        "forward_clv_probability"
    ]
    >=
    0
).astype(
    int
)


# ============================================================
# SUMMARY — SPREADS
# ============================================================

print()

print(
    "=" * 115
)

print(
    "EVENT-LEVEL SPREAD STALE-LINE TEST"
)

print(
    "=" * 115
)


spread_summary_rows = []


for threshold in SPREAD_THRESHOLDS:

    subset = spread_event_signals[

        spread_event_signals[
            "spread_initial_abs_deviation"
        ]
        >=
        threshold

    ].copy()


    if len(
        subset
    ) == 0:

        continue


    spread_summary_rows.append(
        {
            "minimum_disagreement":
                f"{threshold:.1f}+ pts",

            "signals":
                len(
                    subset
                ),

            "unique_games":
                subset[
                    "odds_api_event_id"
                ].nunique(),

            "positive_clv_rate":
                subset[
                    "positive_forward_clv"
                ].mean(),

            "nonnegative_clv_rate":
                subset[
                    "nonnegative_forward_clv"
                ].mean(),

            "mean_initial_disagreement":
                subset[
                    "spread_initial_abs_deviation"
                ].mean(),

            "mean_forward_clv":
                subset[
                    "forward_clv_points"
                ].mean(),

            "median_forward_clv":
                subset[
                    "forward_clv_points"
                ].median()
        }
    )


spread_summary = pd.DataFrame(
    spread_summary_rows
)


if len(
    spread_summary
) == 0:

    print(
        "Not enough spread data."
    )


else:

    print(

        spread_summary.to_string(

            index=False,

            formatters={

                "positive_clv_rate":
                    lambda x:
                        f"{x:.2%}",

                "nonnegative_clv_rate":
                    lambda x:
                        f"{x:.2%}",

                "mean_initial_disagreement":
                    lambda x:
                        f"{x:.2f}",

                "mean_forward_clv":
                    lambda x:
                        f"{x:+.2f}",

                "median_forward_clv":
                    lambda x:
                        f"{x:+.2f}"
            }
        )
    )


# ============================================================
# SUMMARY — MONEYLINE
# ============================================================

print()

print(
    "=" * 115
)

print(
    "EVENT-LEVEL MONEYLINE STALE-LINE TEST"
)

print(
    "=" * 115
)


moneyline_summary_rows = []


for threshold_pp in MONEYLINE_THRESHOLDS_PP:

    subset = moneyline_event_signals[

        (
            moneyline_event_signals[
                "moneyline_initial_abs_deviation"
            ]
            *
            100.0
        )
        >=
        threshold_pp

    ].copy()


    if len(
        subset
    ) == 0:

        continue


    moneyline_summary_rows.append(
        {
            "minimum_disagreement":
                f"{threshold_pp:.1f}+ pp",

            "signals":
                len(
                    subset
                ),

            "unique_games":
                subset[
                    "odds_api_event_id"
                ].nunique(),

            "positive_clv_rate":
                subset[
                    "positive_forward_clv"
                ].mean(),

            "nonnegative_clv_rate":
                subset[
                    "nonnegative_forward_clv"
                ].mean(),

            "mean_initial_disagreement_pp":
                (
                    subset[
                        "moneyline_initial_abs_deviation"
                    ].mean()
                    *
                    100.0
                ),

            "mean_forward_clv_pp":
                subset[
                    "forward_clv_pp"
                ].mean(),

            "median_forward_clv_pp":
                subset[
                    "forward_clv_pp"
                ].median()
        }
    )


moneyline_summary = pd.DataFrame(
    moneyline_summary_rows
)


if len(
    moneyline_summary
) == 0:

    print(
        "Not enough moneyline data."
    )


else:

    print(

        moneyline_summary.to_string(

            index=False,

            formatters={

                "positive_clv_rate":
                    lambda x:
                        f"{x:.2%}",

                "nonnegative_clv_rate":
                    lambda x:
                        f"{x:.2%}",

                "mean_initial_disagreement_pp":
                    lambda x:
                        f"{x:.2f}",

                "mean_forward_clv_pp":
                    lambda x:
                        f"{x:+.2f}",

                "median_forward_clv_pp":
                    lambda x:
                        f"{x:+.2f}"
            }
        )
    )


# ============================================================
# CONSERVATIVE TEST:
#
# FIRST QUALIFYING SIGNAL PER GAME
#
# This prevents one game from repeatedly generating signals
# across several snapshot intervals and dominating the sample.
# ============================================================

print()

print(
    "=" * 115
)

print(
    "FIRST 1+ POINT SPREAD SIGNAL PER GAME"
)

print(
    "=" * 115
)


first_spread = spread_event_signals[

    spread_event_signals[
        "spread_initial_abs_deviation"
    ]
    >=
    1.0

].copy()


first_spread[
    "current_datetime"
] = pd.to_datetime(

    first_spread[
        "current_time"
    ],

    utc=True,

    errors="coerce"
)


first_spread = (

    first_spread

    .sort_values(
        "current_datetime"
    )

    .drop_duplicates(
        "odds_api_event_id",
        keep="first"
    )
)


if len(
    first_spread
) == 0:

    print(
        "No qualifying spread signals."
    )


else:

    print(
        f"Signals:              "
        f"{len(first_spread):,}"
    )


    print(
        f"Positive forward CLV: "
        f"{first_spread['positive_forward_clv'].mean():.2%}"
    )


    print(
        f"Nonnegative CLV:      "
        f"{first_spread['nonnegative_forward_clv'].mean():.2%}"
    )


    print(
        f"Mean forward CLV:     "
        f"{first_spread['forward_clv_points'].mean():+.2f} points"
    )


    print(
        f"Median forward CLV:   "
        f"{first_spread['forward_clv_points'].median():+.2f} points"
    )


print()

print(
    "=" * 115
)

print(
    "FIRST 1+ PP MONEYLINE SIGNAL PER GAME"
)

print(
    "=" * 115
)


first_moneyline = moneyline_event_signals[

    (
        moneyline_event_signals[
            "moneyline_initial_abs_deviation"
        ]
        *
        100.0
    )
    >=
    1.0

].copy()


first_moneyline[
    "current_datetime"
] = pd.to_datetime(

    first_moneyline[
        "current_time"
    ],

    utc=True,

    errors="coerce"
)


first_moneyline = (

    first_moneyline

    .sort_values(
        "current_datetime"
    )

    .drop_duplicates(
        "odds_api_event_id",
        keep="first"
    )
)


if len(
    first_moneyline
) == 0:

    print(
        "No qualifying moneyline signals."
    )


else:

    print(
        f"Signals:              "
        f"{len(first_moneyline):,}"
    )


    print(
        f"Positive forward CLV: "
        f"{first_moneyline['positive_forward_clv'].mean():.2%}"
    )


    print(
        f"Nonnegative CLV:      "
        f"{first_moneyline['nonnegative_forward_clv'].mean():.2%}"
    )


    print(
        f"Mean forward CLV:     "
        f"{first_moneyline['forward_clv_pp'].mean():+.2f} pp"
    )


    print(
        f"Median forward CLV:   "
        f"{first_moneyline['forward_clv_pp'].median():+.2f} pp"
    )


# ============================================================
# LARGEST CURRENT RESULTS
# ============================================================

print()

print(
    "=" * 115
)

print(
    "LARGEST EVENT-LEVEL SPREAD SIGNALS"
)

print(
    "=" * 115
)


largest_spreads = (

    spread_event_signals

    .sort_values(
        "spread_initial_abs_deviation",
        ascending=False
    )

    .head(
        20
    )
)


print(

    largest_spreads[
        [
            "away_team",
            "home_team",
            "bookmaker_key",
            "spread_target_before",
            "spread_consensus_before",
            "spread_consensus_after",
            "stale_side",
            "spread_initial_abs_deviation",
            "forward_clv_points",
            "hours_until_kickoff"
        ]
    ]

    .to_string(

        index=False,

        formatters={

            "spread_target_before":
                lambda x:
                    f"{x:+.1f}",

            "spread_consensus_before":
                lambda x:
                    f"{x:+.1f}",

            "spread_consensus_after":
                lambda x:
                    f"{x:+.1f}",

            "spread_initial_abs_deviation":
                lambda x:
                    f"{x:.1f}",

            "forward_clv_points":
                lambda x:
                    f"{x:+.1f}",

            "hours_until_kickoff":
                lambda x:
                    f"{x:.1f}"
        }
    )
)


# ============================================================
# COMBINE + SAVE
# ============================================================

spread_save = spread_event_signals.copy()

spread_save[
    "market_type"
] = "spread"


moneyline_save = moneyline_event_signals.copy()

moneyline_save[
    "market_type"
] = "moneyline"


combined = pd.concat(

    [
        spread_save,
        moneyline_save
    ],

    ignore_index=True,

    sort=False
)


os.makedirs(
    "data/processed",
    exist_ok=True
)


combined.to_csv(
    OUTPUT_FILE,
    index=False
)


print()

print(
    "=" * 115
)

print(
    "ANALYSIS COMPLETE"
)

print(
    "=" * 115
)


print(
    OUTPUT_FILE
)