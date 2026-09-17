import os
import sqlite3

import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

DATABASE_FILE = "data/cfb.db"

OUTPUT_DIRECTORY = "data/processed"


EXECUTION_BOOK = (
    "hardrockbet_fl"
)


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


def deviation_bucket_probability(
    value
):

    value = abs(
        value
    ) * 100.0


    if value < 1.0:

        return "<1 pp"

    if value < 2.0:

        return "1-2 pp"

    if value < 3.0:

        return "2-3 pp"

    if value < 5.0:

        return "3-5 pp"

    return "5+ pp"


def deviation_bucket_spread(
    value
):

    value = abs(
        value
    )


    if value < 0.5:

        return "<0.5"

    if value < 1.0:

        return "0.5-0.99"

    if value < 1.5:

        return "1.0-1.49"

    if value < 2.5:

        return "1.5-2.49"

    return "2.5+"


# ============================================================
# LOAD DATA
# ============================================================

print(
    "MARKET CONSENSUS CONVERGENCE ANALYSIS — V1"
)

print(
    "=" * 110
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
        "No pregame market snapshots were found."
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
# DE-VIG MONEYLINES
#
# Example:
#
# home raw implied = 60%
# away raw implied = 44%
#
# total = 104%
#
# fair home =
#
# 60 / 104 = 57.69%
#
# This removes the book's margin before comparing books.
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
    "raw_probability_sum"
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
        "raw_probability_sum"
    ]
)


# ============================================================
# LEAVE-ONE-OUT CONSENSUS
#
# This is critical.
#
# If we are evaluating DraftKings, DraftKings is NOT allowed
# to help create the consensus used to judge DraftKings.
#
# Consensus = median of the OTHER reference books.
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


        reference = group[

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


        probability_values = (

            reference[
                "fair_home_probability"
            ]

            .dropna()
        )


        spread_values = (

            reference[
                "home_spread"
            ]

            .dropna()
        )


        if len(
            probability_values
        ) > 0:

            probability_consensus = float(

                probability_values.median()
            )


        else:

            probability_consensus = np.nan


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
        ] = probability_consensus


        row[
            "moneyline_consensus_books"
        ] = len(
            probability_values
        )


        row[
            "spread_consensus"
        ] = spread_consensus


        row[
            "spread_consensus_books"
        ] = len(
            spread_values
        )


        row[
            "moneyline_deviation"
        ] = (

            target[
                "fair_home_probability"
            ]

            -

            probability_consensus

            if (
                pd.notna(
                    target[
                        "fair_home_probability"
                    ]
                )

                and

                pd.notna(
                    probability_consensus
                )
            )

            else
            np.nan
        )


        row[
            "spread_deviation"
        ] = (

            target[
                "home_spread"
            ]

            -

            spread_consensus

            if (
                pd.notna(
                    target[
                        "home_spread"
                    ]
                )

                and

                pd.notna(
                    spread_consensus
                )
            )

            else
            np.nan
        )


        consensus_rows.append(
            row
        )


consensus = pd.DataFrame(
    consensus_rows
)


# ============================================================
# BUILD CONSECUTIVE SNAPSHOT TRANSITIONS
#
# Question:
#
# If Book X disagreed with consensus at snapshot T,
# was Book X closer to that OLD consensus at snapshot T+1?
#
# We use the old consensus deliberately.
#
# That tells us whether the market consensus predicted the
# target book's future direction.
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


        row = {

            "odds_api_event_id":
                event_id,

            "bookmaker_key":
                bookmaker_key,

            "bookmaker_title":
                current[
                    "bookmaker_title"
                ],

            "home_team":
                current[
                    "home_team"
                ],

            "away_team":
                current[
                    "away_team"
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
        # MONEYLINE
        # ====================================================

        if (

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
        ):

            old_distance = abs(

                current[
                    "fair_home_probability"
                ]

                -

                current[
                    "moneyline_consensus"
                ]
            )


            new_distance_to_old_consensus = abs(

                following[
                    "fair_home_probability"
                ]

                -

                current[
                    "moneyline_consensus"
                ]
            )


            row[
                "moneyline_consensus_books"
            ] = current[
                "moneyline_consensus_books"
            ]


            row[
                "moneyline_initial_deviation"
            ] = current[
                "moneyline_deviation"
            ]


            row[
                "moneyline_abs_deviation_before"
            ] = old_distance


            row[
                "moneyline_abs_distance_after"
            ] = new_distance_to_old_consensus


            row[
                "moneyline_improvement"
            ] = (

                old_distance

                -

                new_distance_to_old_consensus
            )


            row[
                "moneyline_moved_toward_consensus"
            ] = int(

                new_distance_to_old_consensus

                <
                old_distance
            )


        else:

            row[
                "moneyline_consensus_books"
            ] = np.nan


            row[
                "moneyline_initial_deviation"
            ] = np.nan


            row[
                "moneyline_abs_deviation_before"
            ] = np.nan


            row[
                "moneyline_abs_distance_after"
            ] = np.nan


            row[
                "moneyline_improvement"
            ] = np.nan


            row[
                "moneyline_moved_toward_consensus"
            ] = np.nan


        # ====================================================
        # SPREAD
        # ====================================================

        if (

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
        ):

            old_distance = abs(

                current[
                    "home_spread"
                ]

                -

                current[
                    "spread_consensus"
                ]
            )


            new_distance_to_old_consensus = abs(

                following[
                    "home_spread"
                ]

                -

                current[
                    "spread_consensus"
                ]
            )


            row[
                "spread_consensus_books"
            ] = current[
                "spread_consensus_books"
            ]


            row[
                "spread_initial_deviation"
            ] = current[
                "spread_deviation"
            ]


            row[
                "spread_abs_deviation_before"
            ] = old_distance


            row[
                "spread_abs_distance_after"
            ] = new_distance_to_old_consensus


            row[
                "spread_improvement"
            ] = (

                old_distance

                -

                new_distance_to_old_consensus
            )


            row[
                "spread_moved_toward_consensus"
            ] = int(

                new_distance_to_old_consensus

                <
                old_distance
            )


        else:

            row[
                "spread_consensus_books"
            ] = np.nan


            row[
                "spread_initial_deviation"
            ] = np.nan


            row[
                "spread_abs_deviation_before"
            ] = np.nan


            row[
                "spread_abs_distance_after"
            ] = np.nan


            row[
                "spread_improvement"
            ] = np.nan


            row[
                "spread_moved_toward_consensus"
            ] = np.nan


        transition_rows.append(
            row
        )


transitions = pd.DataFrame(
    transition_rows
)


print()

print(
    "=" * 110
)

print(
    "TRANSITION COVERAGE"
)

print(
    "=" * 110
)


if len(
    transitions
) == 0:

    print(
        "No consecutive pregame snapshots yet."
    )


    print(
        "Run the market collector again later."
    )


else:

    transition_coverage = (

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
                "moneyline_moved_toward_consensus",
                lambda x:
                    x.notna().sum()
            ),

            spread_tests=(
                "spread_moved_toward_consensus",
                lambda x:
                    x.notna().sum()
            )
        )

        .reset_index()

        .sort_values(
            "transitions",
            ascending=False
        )
    )


    print(
        transition_coverage.to_string(
            index=False
        )
    )


# ============================================================
# THRESHOLD SUMMARIES
#
# These are more useful than one giant overall percentage.
#
# We want to know whether LARGE deviations are more likely to
# converge than tiny differences.
# ============================================================

def print_moneyline_thresholds(
    frame,
    minimum_consensus_books,
    title
):

    print()

    print(
        title
    )

    print(
        "-" * 100
    )


    rows = []


    for threshold_pp in [

        0.0,
        1.0,
        2.0,
        3.0,
        5.0
    ]:

        subset = frame[

            frame[
                "moneyline_moved_toward_consensus"
            ].notna()

            &

            (
                frame[
                    "moneyline_consensus_books"
                ]
                >=
                minimum_consensus_books
            )

            &

            (
                frame[
                    "moneyline_abs_deviation_before"
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


        rows.append(
            {
                "minimum_deviation":
                    f"{threshold_pp:.0f}+ pp",

                "transitions":
                    len(
                        subset
                    ),

                "toward_consensus_rate":
                    subset[
                        "moneyline_moved_toward_consensus"
                    ].mean(),

                "mean_starting_deviation_pp":
                    (
                        subset[
                            "moneyline_abs_deviation_before"
                        ].mean()
                        *
                        100.0
                    ),

                "mean_improvement_pp":
                    (
                        subset[
                            "moneyline_improvement"
                        ].mean()
                        *
                        100.0
                    )
            }
        )


    if len(
        rows
    ) == 0:

        print(
            "Not enough data yet."
        )

        return


    result = pd.DataFrame(
        rows
    )


    print(

        result.to_string(

            index=False,

            formatters={

                "toward_consensus_rate":
                    lambda x:
                        f"{x:.2%}",

                "mean_starting_deviation_pp":
                    lambda x:
                        f"{x:.2f}",

                "mean_improvement_pp":
                    lambda x:
                        f"{x:+.2f}"
            }
        )
    )


def print_spread_thresholds(
    frame,
    minimum_consensus_books,
    title
):

    print()

    print(
        title
    )

    print(
        "-" * 100
    )


    rows = []


    for threshold in [

        0.0,
        0.5,
        1.0,
        1.5,
        2.5
    ]:

        subset = frame[

            frame[
                "spread_moved_toward_consensus"
            ].notna()

            &

            (
                frame[
                    "spread_consensus_books"
                ]
                >=
                minimum_consensus_books
            )

            &

            (
                frame[
                    "spread_abs_deviation_before"
                ]
                >=
                threshold
            )
        ]


        if len(
            subset
        ) == 0:

            continue


        rows.append(
            {
                "minimum_deviation":
                    f"{threshold:.1f}+ pts",

                "transitions":
                    len(
                        subset
                    ),

                "toward_consensus_rate":
                    subset[
                        "spread_moved_toward_consensus"
                    ].mean(),

                "mean_starting_deviation":
                    subset[
                        "spread_abs_deviation_before"
                    ].mean(),

                "mean_improvement":
                    subset[
                        "spread_improvement"
                    ].mean()
            }
        )


    if len(
        rows
    ) == 0:

        print(
            "Not enough data yet."
        )

        return


    result = pd.DataFrame(
        rows
    )


    print(

        result.to_string(

            index=False,

            formatters={

                "toward_consensus_rate":
                    lambda x:
                        f"{x:.2%}",

                "mean_starting_deviation":
                    lambda x:
                        f"{x:.2f}",

                "mean_improvement":
                    lambda x:
                        f"{x:+.2f}"
            }
        )
    )


if len(
    transitions
) > 0:

    print_moneyline_thresholds(

        transitions,

        minimum_consensus_books=1,

        title=(
            "MONEYLINE CONVERGENCE — "
            "ALL AVAILABLE CONSENSUS"
        )
    )


    print_spread_thresholds(

        transitions,

        minimum_consensus_books=1,

        title=(
            "SPREAD CONVERGENCE — "
            "ALL AVAILABLE CONSENSUS"
        )
    )


    print_moneyline_thresholds(

        transitions,

        minimum_consensus_books=3,

        title=(
            "MONEYLINE CONVERGENCE — "
            "3+ OTHER BOOKS"
        )
    )


    print_spread_thresholds(

        transitions,

        minimum_consensus_books=3,

        title=(
            "SPREAD CONVERGENCE — "
            "3+ OTHER BOOKS"
        )
    )


# ============================================================
# LATEST CURRENT OUTLIERS
# ============================================================

latest_time = consensus[
    "captured_datetime"
].max()


latest = consensus[

    consensus[
        "captured_datetime"
    ]
    ==
    latest_time

].copy()


latest[
    "moneyline_deviation_pp"
] = (

    latest[
        "moneyline_deviation"
    ]

    *
    100.0
)


latest[
    "outlier_score"
] = (

    latest[
        "moneyline_deviation_pp"
    ].abs().fillna(
        0.0
    )

    +

    latest[
        "spread_deviation"
    ].abs().fillna(
        0.0
    )
)


current_outliers = (

    latest[

        (
            latest[
                "moneyline_consensus_books"
            ]
            >=
            3
        )

        |

        (
            latest[
                "spread_consensus_books"
            ]
            >=
            3
        )
    ]

    .sort_values(
        "outlier_score",
        ascending=False
    )

    .head(
        30
    )
)


print()

print(
    "=" * 110
)

print(
    "CURRENT MARKET OUTLIERS"
)

print(
    "=" * 110
)


if len(
    current_outliers
) == 0:

    print(
        "No games currently have 3+ comparison books."
    )


else:

    print(

        current_outliers[
            [
                "away_team",
                "home_team",
                "bookmaker_key",
                "home_spread",
                "spread_consensus",
                "spread_deviation",
                "fair_home_probability",
                "moneyline_consensus",
                "moneyline_deviation_pp",
                "hours_until_kickoff"
            ]
        ]

        .to_string(

            index=False,

            formatters={

                "home_spread":
                    lambda x:
                        f"{x:+.1f}",

                "spread_consensus":
                    lambda x:
                        f"{x:+.1f}",

                "spread_deviation":
                    lambda x:
                        f"{x:+.1f}",

                "fair_home_probability":
                    lambda x:
                        f"{x:.2%}",

                "moneyline_consensus":
                    lambda x:
                        f"{x:.2%}",

                "moneyline_deviation_pp":
                    lambda x:
                        f"{x:+.2f}",

                "hours_until_kickoff":
                    lambda x:
                        f"{x:.1f}"
            }
        )
    )


# ============================================================
# HARD ROCK STATUS
# ============================================================

hard_rock = latest[

    latest[
        "bookmaker_key"
    ]
    ==
    EXECUTION_BOOK

]


print()

print(
    "=" * 110
)

print(
    "HARD ROCK FL STATUS"
)

print(
    "=" * 110
)


if len(
    hard_rock
) == 0:

    print(
        "Hard Rock FL NCAAF is still absent "
        "from the current API snapshot."
    )


else:

    print(
        f"Hard Rock games available: "
        f"{len(hard_rock):,}"
    )


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    OUTPUT_DIRECTORY,
    exist_ok=True
)


consensus.to_csv(

    os.path.join(
        OUTPUT_DIRECTORY,
        "live_market_consensus_v1.csv"
    ),

    index=False
)


transitions.to_csv(

    os.path.join(
        OUTPUT_DIRECTORY,
        "live_market_convergence_v1.csv"
    ),

    index=False
)


print()

print(
    "=" * 110
)

print(
    "ANALYSIS COMPLETE"
)

print(
    "=" * 110
)


print(
    "data/processed/live_market_consensus_v1.csv"
)


print(
    "data/processed/live_market_convergence_v1.csv"
)