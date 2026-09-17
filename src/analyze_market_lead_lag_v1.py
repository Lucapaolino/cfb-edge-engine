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
    "market_lead_lag_transitions_v1.csv"
)


EXECUTION_BOOK = "hardrockbet_fl"


REFERENCE_BOOKS = [

    "draftkings",
    "fanduel",
    "betmgm",
    "betrivers",
    "bovada",
    "betonlineag",
    "betus",
    "pinnacle",
    "espnbet"
]


# ============================================================
# ANALYSIS REQUIREMENTS
#
# A consensus must contain at least 3 OTHER sportsbooks at
# BOTH snapshots.
#
# We also reject transitions separated by more than 36 hours.
# That prevents a three-day gap from being treated like a
# normal sequential market update.
# ============================================================

MINIMUM_CONSENSUS_BOOKS = 3

MAX_HOURS_BETWEEN = 36.0


# ============================================================
# MOVEMENT TOLERANCE
#
# Moneyline:
# 0.0025 probability = 0.25 percentage points.
#
# Spread:
# 0.25 points.
#
# This stops microscopic numerical changes from being called
# real market movement.
# ============================================================

MONEYLINE_MOVEMENT_TOLERANCE = 0.0025

SPREAD_MOVEMENT_TOLERANCE = 0.25


# ============================================================
# HELPERS
# ============================================================

def american_to_probability(odds):

    if pd.isna(odds):

        return np.nan


    odds = float(
        odds
    )


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


def classify_movement(
    target_moved,
    consensus_moved
):

    if (
        target_moved
        and
        consensus_moved
    ):

        return "BOTH_MOVED"


    if target_moved:

        return "TARGET_CONVERGED"


    if consensus_moved:

        return "CONSENSUS_CONVERGED"


    return "NEITHER_MOVED"


# ============================================================
# LOAD DATA
# ============================================================

print(
    "MARKET LEAD / LAG ANALYSIS — V1"
)

print(
    "=" * 115
)


connection = sqlite3.connect(
    DATABASE_FILE
)


data = pd.read_sql_query(

    """
    SELECT *

    FROM execution_market_snapshots

    WHERE is_pregame = 1

    ORDER BY
        captured_at_utc,
        odds_api_event_id,
        bookmaker_key
    """,

    connection
)


connection.close()


if len(
    data
) == 0:

    raise RuntimeError(
        "No pregame execution-market snapshots found."
    )


data[
    "captured_datetime"
] = pd.to_datetime(

    data[
        "captured_at_utc"
    ],

    utc=True,

    errors="coerce"
)


numeric_columns = [

    "home_moneyline",
    "away_moneyline",
    "home_spread",
    "away_spread",
    "hours_until_kickoff"
]


for column in numeric_columns:

    data[
        column
    ] = pd.to_numeric(

        data[
            column
        ],

        errors="coerce"
    )


print(
    f"Pregame rows:      "
    f"{len(data):,}"
)


print(
    f"Unique games:      "
    f"{data['odds_api_event_id'].nunique():,}"
)


print(
    f"Distinct batches:  "
    f"{data['batch_id'].nunique():,}"
)


print(
    f"Books represented: "
    f"{data['bookmaker_key'].nunique():,}"
)


# ============================================================
# DE-VIG MONEYLINE
# ============================================================

data[
    "home_raw_probability"
] = data[
    "home_moneyline"
].apply(
    american_to_probability
)


data[
    "away_raw_probability"
] = data[
    "away_moneyline"
].apply(
    american_to_probability
)


data[
    "probability_sum"
] = (

    data[
        "home_raw_probability"
    ]

    +

    data[
        "away_raw_probability"
    ]
)


data[
    "fair_home_probability"
] = (

    data[
        "home_raw_probability"
    ]

    /

    data[
        "probability_sum"
    ]
)


# ============================================================
# BASIC MONEYLINE INTEGRITY
#
# Extremely malformed markets are not allowed into consensus.
#
# Normal sportsbook vig is usually a little above 1.00.
# We allow a broad 1.00 to 1.15 range.
# ============================================================

data[
    "valid_moneyline"
] = (

    data[
        "home_raw_probability"
    ].notna()

    &

    data[
        "away_raw_probability"
    ].notna()

    &

    (
        data[
            "probability_sum"
        ]
        >=
        1.00
    )

    &

    (
        data[
            "probability_sum"
        ]
        <=
        1.15
    )
)


data.loc[
    ~data[
        "valid_moneyline"
    ],
    "fair_home_probability"
] = np.nan


# ============================================================
# LEAVE-ONE-BOOK-OUT CONSENSUS
#
# Example:
#
# If evaluating FanDuel:
#
# FanDuel itself is removed.
#
# Consensus is the median of the remaining reference books.
#
# This avoids judging FanDuel partly against FanDuel.
# ============================================================

consensus_rows = []


for (
    batch_id,
    event_id
), group in data.groupby(
    [
        "batch_id",
        "odds_api_event_id"
    ]
):

    group = group.copy()


    for _, target in group.iterrows():

        target_book = target[
            "bookmaker_key"
        ]


        references = group[

            group[
                "bookmaker_key"
            ].isin(
                REFERENCE_BOOKS
            )

            &

            (
                group[
                    "bookmaker_key"
                ]
                !=
                target_book
            )
        ].copy()


        moneyline_values = (

            references[
                "fair_home_probability"
            ]

            .dropna()
        )


        spread_values = (

            references[
                "home_spread"
            ]

            .dropna()
        )


        if len(
            moneyline_values
        ) > 0:

            moneyline_consensus = float(

                moneyline_values.median()
            )


        else:

            moneyline_consensus = np.nan


        if len(
            spread_values
        ) > 0:

            spread_consensus = float(

                spread_values.median()
            )


        else:

            spread_consensus = np.nan


        row = target.to_dict()


        row[
            "moneyline_consensus"
        ] = moneyline_consensus


        row[
            "moneyline_consensus_books"
        ] = len(
            moneyline_values
        )


        row[
            "spread_consensus"
        ] = spread_consensus


        row[
            "spread_consensus_books"
        ] = len(
            spread_values
        )


        consensus_rows.append(
            row
        )


consensus = pd.DataFrame(
    consensus_rows
)


# ============================================================
# CONSECUTIVE OBSERVATIONS
#
# At Time 0:
#
# target = X0
# consensus = C0
#
# At Time 1:
#
# target = X1
# consensus = C1
#
#
# TARGET CONVERGED:
#
# X1 moved closer to OLD consensus C0.
#
#
# CONSENSUS CONVERGED:
#
# C1 moved closer to OLD target X0.
#
#
# BOTH MOVED:
#
# both moved toward the other's old position.
#
#
# NEITHER:
#
# neither made a meaningful move toward the other.
# ============================================================

transition_rows = []


for (
    event_id,
    bookmaker_key
), group in consensus.groupby(
    [
        "odds_api_event_id",
        "bookmaker_key"
    ]
):

    group = (

        group

        .sort_values(
            "captured_datetime"
        )

        .reset_index(
            drop=True
        )
    )


    if len(
        group
    ) < 2:

        continue


    for position in range(
        len(group) - 1
    ):

        current = group.iloc[
            position
        ]


        following = group.iloc[
            position + 1
        ]


        hours_between = (

            following[
                "captured_datetime"
            ]

            -

            current[
                "captured_datetime"
            ]

        ).total_seconds() / 3600.0


        # ----------------------------------------------------
        # Skip weird/non-sequential gaps.
        # ----------------------------------------------------

        if (
            hours_between <= 0

            or

            hours_between
            >
            MAX_HOURS_BETWEEN
        ):

            continue


        result = {

            "odds_api_event_id":
                event_id,

            "bookmaker_key":
                bookmaker_key,

            "bookmaker_title":
                current[
                    "bookmaker_title"
                ],

            "away_team":
                current[
                    "away_team"
                ],

            "home_team":
                current[
                    "home_team"
                ],

            "current_time":
                current[
                    "captured_at_utc"
                ],

            "next_time":
                following[
                    "captured_at_utc"
                ],

            "hours_between":
                hours_between,

            "hours_until_kickoff":
                current[
                    "hours_until_kickoff"
                ]
        }


        # ====================================================
        # MONEYLINE LEAD / LAG
        # ====================================================

        moneyline_usable = (

            pd.notna(
                current[
                    "fair_home_probability"
                ]
            )

            and

            pd.notna(
                following[
                    "fair_home_probability"
                ]
            )

            and

            pd.notna(
                current[
                    "moneyline_consensus"
                ]
            )

            and

            pd.notna(
                following[
                    "moneyline_consensus"
                ]
            )

            and

            current[
                "moneyline_consensus_books"
            ]
            >=
            MINIMUM_CONSENSUS_BOOKS

            and

            following[
                "moneyline_consensus_books"
            ]
            >=
            MINIMUM_CONSENSUS_BOOKS
        )


        if moneyline_usable:

            target_0 = float(

                current[
                    "fair_home_probability"
                ]
            )


            consensus_0 = float(

                current[
                    "moneyline_consensus"
                ]
            )


            target_1 = float(

                following[
                    "fair_home_probability"
                ]
            )


            consensus_1 = float(

                following[
                    "moneyline_consensus"
                ]
            )


            initial_gap = (

                target_0
                -
                consensus_0
            )


            initial_distance = abs(
                initial_gap
            )


            final_distance = abs(

                target_1
                -
                consensus_1
            )


            # ------------------------------------------------
            # Did target move toward OLD consensus?
            # ------------------------------------------------

            target_distance_to_old_consensus = abs(

                target_1
                -
                consensus_0
            )


            target_progress = (

                initial_distance

                -

                target_distance_to_old_consensus
            )


            target_moved = (

                target_progress

                >
                MONEYLINE_MOVEMENT_TOLERANCE
            )


            # ------------------------------------------------
            # Did consensus move toward OLD target?
            # ------------------------------------------------

            consensus_distance_to_old_target = abs(

                consensus_1
                -
                target_0
            )


            consensus_progress = (

                initial_distance

                -

                consensus_distance_to_old_target
            )


            consensus_moved = (

                consensus_progress

                >
                MONEYLINE_MOVEMENT_TOLERANCE
            )


            classification = classify_movement(

                target_moved,

                consensus_moved
            )


            result[
                "moneyline_consensus_books"
            ] = current[
                "moneyline_consensus_books"
            ]


            result[
                "moneyline_target_before"
            ] = target_0


            result[
                "moneyline_consensus_before"
            ] = consensus_0


            result[
                "moneyline_target_after"
            ] = target_1


            result[
                "moneyline_consensus_after"
            ] = consensus_1


            result[
                "moneyline_initial_deviation"
            ] = initial_gap


            result[
                "moneyline_initial_abs_deviation"
            ] = initial_distance


            result[
                "moneyline_final_abs_deviation"
            ] = final_distance


            result[
                "moneyline_gap_closed"
            ] = (

                initial_distance

                -

                final_distance
            )


            result[
                "moneyline_target_progress"
            ] = target_progress


            result[
                "moneyline_consensus_progress"
            ] = consensus_progress


            result[
                "moneyline_target_move"
            ] = (

                target_1
                -
                target_0
            )


            result[
                "moneyline_consensus_move"
            ] = (

                consensus_1
                -
                consensus_0
            )


            result[
                "moneyline_classification"
            ] = classification


        else:

            result[
                "moneyline_consensus_books"
            ] = np.nan


            result[
                "moneyline_target_before"
            ] = np.nan


            result[
                "moneyline_consensus_before"
            ] = np.nan


            result[
                "moneyline_target_after"
            ] = np.nan


            result[
                "moneyline_consensus_after"
            ] = np.nan


            result[
                "moneyline_initial_deviation"
            ] = np.nan


            result[
                "moneyline_initial_abs_deviation"
            ] = np.nan


            result[
                "moneyline_final_abs_deviation"
            ] = np.nan


            result[
                "moneyline_gap_closed"
            ] = np.nan


            result[
                "moneyline_target_progress"
            ] = np.nan


            result[
                "moneyline_consensus_progress"
            ] = np.nan


            result[
                "moneyline_target_move"
            ] = np.nan


            result[
                "moneyline_consensus_move"
            ] = np.nan


            result[
                "moneyline_classification"
            ] = None


        # ====================================================
        # SPREAD LEAD / LAG
        # ====================================================

        spread_usable = (

            pd.notna(
                current[
                    "home_spread"
                ]
            )

            and

            pd.notna(
                following[
                    "home_spread"
                ]
            )

            and

            pd.notna(
                current[
                    "spread_consensus"
                ]
            )

            and

            pd.notna(
                following[
                    "spread_consensus"
                ]
            )

            and

            current[
                "spread_consensus_books"
            ]
            >=
            MINIMUM_CONSENSUS_BOOKS

            and

            following[
                "spread_consensus_books"
            ]
            >=
            MINIMUM_CONSENSUS_BOOKS
        )


        if spread_usable:

            target_0 = float(

                current[
                    "home_spread"
                ]
            )


            consensus_0 = float(

                current[
                    "spread_consensus"
                ]
            )


            target_1 = float(

                following[
                    "home_spread"
                ]
            )


            consensus_1 = float(

                following[
                    "spread_consensus"
                ]
            )


            initial_gap = (

                target_0
                -
                consensus_0
            )


            initial_distance = abs(
                initial_gap
            )


            final_distance = abs(

                target_1
                -
                consensus_1
            )


            # ------------------------------------------------
            # TARGET moves toward OLD consensus
            # ------------------------------------------------

            target_distance_to_old_consensus = abs(

                target_1
                -
                consensus_0
            )


            target_progress = (

                initial_distance

                -

                target_distance_to_old_consensus
            )


            target_moved = (

                target_progress

                >
                SPREAD_MOVEMENT_TOLERANCE
            )


            # ------------------------------------------------
            # CONSENSUS moves toward OLD target
            # ------------------------------------------------

            consensus_distance_to_old_target = abs(

                consensus_1
                -
                target_0
            )


            consensus_progress = (

                initial_distance

                -

                consensus_distance_to_old_target
            )


            consensus_moved = (

                consensus_progress

                >
                SPREAD_MOVEMENT_TOLERANCE
            )


            classification = classify_movement(

                target_moved,

                consensus_moved
            )


            result[
                "spread_consensus_books"
            ] = current[
                "spread_consensus_books"
            ]


            result[
                "spread_target_before"
            ] = target_0


            result[
                "spread_consensus_before"
            ] = consensus_0


            result[
                "spread_target_after"
            ] = target_1


            result[
                "spread_consensus_after"
            ] = consensus_1


            result[
                "spread_initial_deviation"
            ] = initial_gap


            result[
                "spread_initial_abs_deviation"
            ] = initial_distance


            result[
                "spread_final_abs_deviation"
            ] = final_distance


            result[
                "spread_gap_closed"
            ] = (

                initial_distance

                -

                final_distance
            )


            result[
                "spread_target_progress"
            ] = target_progress


            result[
                "spread_consensus_progress"
            ] = consensus_progress


            result[
                "spread_target_move"
            ] = (

                target_1
                -
                target_0
            )


            result[
                "spread_consensus_move"
            ] = (

                consensus_1
                -
                consensus_0
            )


            result[
                "spread_classification"
            ] = classification


        else:

            result[
                "spread_consensus_books"
            ] = np.nan


            result[
                "spread_target_before"
            ] = np.nan


            result[
                "spread_consensus_before"
            ] = np.nan


            result[
                "spread_target_after"
            ] = np.nan


            result[
                "spread_consensus_after"
            ] = np.nan


            result[
                "spread_initial_deviation"
            ] = np.nan


            result[
                "spread_initial_abs_deviation"
            ] = np.nan


            result[
                "spread_final_abs_deviation"
            ] = np.nan


            result[
                "spread_gap_closed"
            ] = np.nan


            result[
                "spread_target_progress"
            ] = np.nan


            result[
                "spread_consensus_progress"
            ] = np.nan


            result[
                "spread_target_move"
            ] = np.nan


            result[
                "spread_consensus_move"
            ] = np.nan


            result[
                "spread_classification"
            ] = None


        transition_rows.append(
            result
        )


transitions = pd.DataFrame(
    transition_rows
)


if len(
    transitions
) == 0:

    raise RuntimeError(
        "No usable sequential transitions found."
    )


# ============================================================
# COVERAGE
# ============================================================

print()

print(
    "=" * 115
)

print(
    "QUALIFIED TRANSITION COVERAGE"
)

print(
    "=" * 115
)


coverage = (

    transitions

    .groupby(
        "bookmaker_key"
    )

    .agg(

        transitions=(
            "odds_api_event_id",
            "size"
        ),

        moneyline_tests=(
            "moneyline_classification",
            lambda x:
                x.notna().sum()
        ),

        spread_tests=(
            "spread_classification",
            lambda x:
                x.notna().sum()
        ),

        mean_hours_between=(
            "hours_between",
            "mean"
        )
    )

    .reset_index()

    .sort_values(
        "transitions",
        ascending=False
    )
)


print(

    coverage.to_string(

        index=False,

        formatters={

            "mean_hours_between":
                lambda x:
                    f"{x:.1f}"
        }
    )
)


# ============================================================
# SUMMARY FUNCTIONS
# ============================================================

def summarize_moneyline_thresholds():

    print()

    print(
        "=" * 115
    )

    print(
        "MONEYLINE LEAD / LAG — 3+ OTHER BOOKS"
    )

    print(
        "=" * 115
    )


    rows = []


    for threshold_pp in [

        0.5,
        1.0,
        1.5,
        2.0,
        3.0
    ]:

        subset = transitions[

            transitions[
                "moneyline_classification"
            ].notna()

            &

            (
                transitions[
                    "moneyline_initial_abs_deviation"
                ]
                *
                100.0

                >=
                threshold_pp
            )
        ]


        if len(
            subset
        ) == 0:

            continue


        classifications = subset[
            "moneyline_classification"
        ]


        rows.append(
            {
                "minimum_deviation":
                    f"{threshold_pp:.1f}+ pp",

                "transitions":
                    len(
                        subset
                    ),

                "target_converged":
                    (
                        classifications
                        ==
                        "TARGET_CONVERGED"
                    ).mean(),

                "consensus_converged":
                    (
                        classifications
                        ==
                        "CONSENSUS_CONVERGED"
                    ).mean(),

                "both_moved":
                    (
                        classifications
                        ==
                        "BOTH_MOVED"
                    ).mean(),

                "neither_moved":
                    (
                        classifications
                        ==
                        "NEITHER_MOVED"
                    ).mean(),

                "gap_closed_rate":
                    (
                        subset[
                            "moneyline_gap_closed"
                        ]
                        >
                        MONEYLINE_MOVEMENT_TOLERANCE
                    ).mean(),

                "mean_gap_closed_pp":
                    (
                        subset[
                            "moneyline_gap_closed"
                        ].mean()
                        *
                        100.0
                    )
            }
        )


    result = pd.DataFrame(
        rows
    )


    if len(
        result
    ) == 0:

        print(
            "Not enough data yet."
        )

        return


    print(

        result.to_string(

            index=False,

            formatters={

                "target_converged":
                    lambda x:
                        f"{x:.2%}",

                "consensus_converged":
                    lambda x:
                        f"{x:.2%}",

                "both_moved":
                    lambda x:
                        f"{x:.2%}",

                "neither_moved":
                    lambda x:
                        f"{x:.2%}",

                "gap_closed_rate":
                    lambda x:
                        f"{x:.2%}",

                "mean_gap_closed_pp":
                    lambda x:
                        f"{x:+.2f}"
            }
        )
    )


def summarize_spread_thresholds():

    print()

    print(
        "=" * 115
    )

    print(
        "SPREAD LEAD / LAG — 3+ OTHER BOOKS"
    )

    print(
        "=" * 115
    )


    rows = []


    for threshold in [

        0.5,
        1.0,
        1.5,
        2.0,
        2.5
    ]:

        subset = transitions[

            transitions[
                "spread_classification"
            ].notna()

            &

            (
                transitions[
                    "spread_initial_abs_deviation"
                ]
                >=
                threshold
            )
        ]


        if len(
            subset
        ) == 0:

            continue


        classifications = subset[
            "spread_classification"
        ]


        rows.append(
            {
                "minimum_deviation":
                    f"{threshold:.1f}+ pts",

                "transitions":
                    len(
                        subset
                    ),

                "target_converged":
                    (
                        classifications
                        ==
                        "TARGET_CONVERGED"
                    ).mean(),

                "consensus_converged":
                    (
                        classifications
                        ==
                        "CONSENSUS_CONVERGED"
                    ).mean(),

                "both_moved":
                    (
                        classifications
                        ==
                        "BOTH_MOVED"
                    ).mean(),

                "neither_moved":
                    (
                        classifications
                        ==
                        "NEITHER_MOVED"
                    ).mean(),

                "gap_closed_rate":
                    (
                        subset[
                            "spread_gap_closed"
                        ]
                        >
                        SPREAD_MOVEMENT_TOLERANCE
                    ).mean(),

                "mean_gap_closed":
                    subset[
                        "spread_gap_closed"
                    ].mean()
            }
        )


    result = pd.DataFrame(
        rows
    )


    if len(
        result
    ) == 0:

        print(
            "Not enough data yet."
        )

        return


    print(

        result.to_string(

            index=False,

            formatters={

                "target_converged":
                    lambda x:
                        f"{x:.2%}",

                "consensus_converged":
                    lambda x:
                        f"{x:.2%}",

                "both_moved":
                    lambda x:
                        f"{x:.2%}",

                "neither_moved":
                    lambda x:
                        f"{x:.2%}",

                "gap_closed_rate":
                    lambda x:
                        f"{x:.2%}",

                "mean_gap_closed":
                    lambda x:
                        f"{x:+.2f}"
            }
        )
    )


summarize_moneyline_thresholds()

summarize_spread_thresholds()


# ============================================================
# BOOK-LEVEL MONEYLINE
#
# Require initial disagreement >= 1 percentage point.
# ============================================================

moneyline_book_data = transitions[

    transitions[
        "moneyline_classification"
    ].notna()

    &

    (
        transitions[
            "moneyline_initial_abs_deviation"
        ]
        >=
        0.01
    )
].copy()


print()

print(
    "=" * 115
)

print(
    "BOOK-LEVEL MONEYLINE LEAD / LAG — 1+ PP"
)

print(
    "=" * 115
)


moneyline_book_rows = []


for book, subset in moneyline_book_data.groupby(
    "bookmaker_key"
):

    classifications = subset[
        "moneyline_classification"
    ]


    moneyline_book_rows.append(
        {
            "bookmaker":
                book,

            "transitions":
                len(
                    subset
                ),

            "target_converged":
                (
                    classifications
                    ==
                    "TARGET_CONVERGED"
                ).mean(),

            "consensus_converged":
                (
                    classifications
                    ==
                    "CONSENSUS_CONVERGED"
                ).mean(),

            "both_moved":
                (
                    classifications
                    ==
                    "BOTH_MOVED"
                ).mean(),

            "neither_moved":
                (
                    classifications
                    ==
                    "NEITHER_MOVED"
                ).mean(),

            "mean_gap_closed_pp":
                (
                    subset[
                        "moneyline_gap_closed"
                    ].mean()
                    *
                    100.0
                )
        }
    )


moneyline_book_summary = pd.DataFrame(
    moneyline_book_rows
)


if len(
    moneyline_book_summary
) == 0:

    print(
        "Not enough data yet."
    )


else:

    moneyline_book_summary = (

        moneyline_book_summary

        .sort_values(
            "transitions",
            ascending=False
        )
    )


    print(

        moneyline_book_summary.to_string(

            index=False,

            formatters={

                "target_converged":
                    lambda x:
                        f"{x:.2%}",

                "consensus_converged":
                    lambda x:
                        f"{x:.2%}",

                "both_moved":
                    lambda x:
                        f"{x:.2%}",

                "neither_moved":
                    lambda x:
                        f"{x:.2%}",

                "mean_gap_closed_pp":
                    lambda x:
                        f"{x:+.2f}"
            }
        )
    )


# ============================================================
# BOOK-LEVEL SPREAD
#
# Require initial disagreement >= 1 point.
# ============================================================

spread_book_data = transitions[

    transitions[
        "spread_classification"
    ].notna()

    &

    (
        transitions[
            "spread_initial_abs_deviation"
        ]
        >=
        1.0
    )
].copy()


print()

print(
    "=" * 115
)

print(
    "BOOK-LEVEL SPREAD LEAD / LAG — 1+ POINT"
)

print(
    "=" * 115
)


spread_book_rows = []


for book, subset in spread_book_data.groupby(
    "bookmaker_key"
):

    classifications = subset[
        "spread_classification"
    ]


    spread_book_rows.append(
        {
            "bookmaker":
                book,

            "transitions":
                len(
                    subset
                ),

            "target_converged":
                (
                    classifications
                    ==
                    "TARGET_CONVERGED"
                ).mean(),

            "consensus_converged":
                (
                    classifications
                    ==
                    "CONSENSUS_CONVERGED"
                ).mean(),

            "both_moved":
                (
                    classifications
                    ==
                    "BOTH_MOVED"
                ).mean(),

            "neither_moved":
                (
                    classifications
                    ==
                    "NEITHER_MOVED"
                ).mean(),

            "mean_gap_closed":
                subset[
                    "spread_gap_closed"
                ].mean()
        }
    )


spread_book_summary = pd.DataFrame(
    spread_book_rows
)


if len(
    spread_book_summary
) == 0:

    print(
        "Not enough data yet."
    )


else:

    spread_book_summary = (

        spread_book_summary

        .sort_values(
            "transitions",
            ascending=False
        )
    )


    print(

        spread_book_summary.to_string(

            index=False,

            formatters={

                "target_converged":
                    lambda x:
                        f"{x:.2%}",

                "consensus_converged":
                    lambda x:
                        f"{x:.2%}",

                "both_moved":
                    lambda x:
                        f"{x:.2%}",

                "neither_moved":
                    lambda x:
                        f"{x:.2%}",

                "mean_gap_closed":
                    lambda x:
                        f"{x:+.2f}"
            }
        )
    )


# ============================================================
# LARGEST SPREAD DISAGREEMENTS
# ============================================================

large_spreads = transitions[

    transitions[
        "spread_classification"
    ].notna()

].copy()


large_spreads = (

    large_spreads

    .sort_values(
        "spread_initial_abs_deviation",
        ascending=False
    )

    .head(
        25
    )
)


print()

print(
    "=" * 115
)

print(
    "LARGEST QUALIFIED SPREAD DISAGREEMENT TRANSITIONS"
)

print(
    "=" * 115
)


if len(
    large_spreads
) == 0:

    print(
        "None."
    )


else:

    print(

        large_spreads[
            [
                "away_team",
                "home_team",
                "bookmaker_key",
                "hours_between",
                "hours_until_kickoff",
                "spread_target_before",
                "spread_consensus_before",
                "spread_target_after",
                "spread_consensus_after",
                "spread_initial_abs_deviation",
                "spread_gap_closed",
                "spread_classification"
            ]
        ].to_string(

            index=False,

            formatters={

                "hours_between":
                    lambda x:
                        f"{x:.1f}",

                "hours_until_kickoff":
                    lambda x:
                        f"{x:.1f}",

                "spread_target_before":
                    lambda x:
                        f"{x:+.1f}",

                "spread_consensus_before":
                    lambda x:
                        f"{x:+.1f}",

                "spread_target_after":
                    lambda x:
                        f"{x:+.1f}",

                "spread_consensus_after":
                    lambda x:
                        f"{x:+.1f}",

                "spread_initial_abs_deviation":
                    lambda x:
                        f"{x:.1f}",

                "spread_gap_closed":
                    lambda x:
                        f"{x:+.1f}"
            }
        )
    )


# ============================================================
# LARGEST MONEYLINE DISAGREEMENTS
# ============================================================

large_moneylines = transitions[

    transitions[
        "moneyline_classification"
    ].notna()

].copy()


large_moneylines[
    "moneyline_initial_abs_deviation_pp"
] = (

    large_moneylines[
        "moneyline_initial_abs_deviation"
    ]

    *
    100.0
)


large_moneylines[
    "moneyline_gap_closed_pp"
] = (

    large_moneylines[
        "moneyline_gap_closed"
    ]

    *
    100.0
)


large_moneylines = (

    large_moneylines

    .sort_values(
        "moneyline_initial_abs_deviation_pp",
        ascending=False
    )

    .head(
        25
    )
)


print()

print(
    "=" * 115
)

print(
    "LARGEST QUALIFIED MONEYLINE DISAGREEMENT TRANSITIONS"
)

print(
    "=" * 115
)


if len(
    large_moneylines
) == 0:

    print(
        "None."
    )


else:

    print(

        large_moneylines[
            [
                "away_team",
                "home_team",
                "bookmaker_key",
                "hours_between",
                "hours_until_kickoff",
                "moneyline_target_before",
                "moneyline_consensus_before",
                "moneyline_target_after",
                "moneyline_consensus_after",
                "moneyline_initial_abs_deviation_pp",
                "moneyline_gap_closed_pp",
                "moneyline_classification"
            ]
        ].to_string(

            index=False,

            formatters={

                "hours_between":
                    lambda x:
                        f"{x:.1f}",

                "hours_until_kickoff":
                    lambda x:
                        f"{x:.1f}",

                "moneyline_target_before":
                    lambda x:
                        f"{x:.2%}",

                "moneyline_consensus_before":
                    lambda x:
                        f"{x:.2%}",

                "moneyline_target_after":
                    lambda x:
                        f"{x:.2%}",

                "moneyline_consensus_after":
                    lambda x:
                        f"{x:.2%}",

                "moneyline_initial_abs_deviation_pp":
                    lambda x:
                        f"{x:.2f}",

                "moneyline_gap_closed_pp":
                    lambda x:
                        f"{x:+.2f}"
            }
        )
    )


# ============================================================
# HARD ROCK
# ============================================================

hard_rock_rows = data[

    data[
        "bookmaker_key"
    ]
    ==
    EXECUTION_BOOK

]


print()

print(
    "=" * 115
)

print(
    "HARD ROCK FL STATUS"
)

print(
    "=" * 115
)


if len(
    hard_rock_rows
) == 0:

    print(
        "Hard Rock FL NCAAF remains unavailable "
        "from this feed."
    )


else:

    print(
        f"Hard Rock pregame rows: "
        f"{len(hard_rock_rows):,}"
    )


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    "data/processed",
    exist_ok=True
)


transitions.to_csv(
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