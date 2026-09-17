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
    "prospective_market_signals_v1.csv"
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


MINIMUM_OTHER_BOOKS = 3

SPREAD_SIGNAL_THRESHOLD = 1.0

MONEYLINE_SIGNAL_THRESHOLD_PP = 1.0


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


def fair_home_probability(
    home_odds,
    away_odds
):

    home_raw = american_to_probability(
        home_odds
    )


    away_raw = american_to_probability(
        away_odds
    )


    if (
        pd.isna(home_raw)
        or
        pd.isna(away_raw)
    ):

        return np.nan


    probability_sum = (

        home_raw
        +
        away_raw
    )


    # Broad integrity guard.
    #
    # Normal two-way markets should generally fall
    # reasonably close to 1.00 after adding both raw
    # implied probabilities.
    #
    # We do NOT reject a market merely because both
    # moneylines are negative.
    if (
        probability_sum < 1.00
        or
        probability_sum > 1.15
    ):

        return np.nan


    return (

        home_raw

        /

        probability_sum
    )


# ============================================================
# START
# ============================================================

print(
    "FREEZE PROSPECTIVE MARKET SIGNALS — V1"
)

print(
    "=" * 115
)


connection = sqlite3.connect(
    DATABASE_FILE
)


# ============================================================
# CREATE PERMANENT SIGNAL LEDGER
#
# signal_id is unique.
#
# Because signal_id is only:
#
#     market_type + game ID
#
# a game can never receive a second spread signal or
# second moneyline signal later.
#
# FIRST QUALIFYING SIGNAL WINS.
# ============================================================

connection.execute(
    """
    CREATE TABLE IF NOT EXISTS
    prospective_market_signals_v1 (

        signal_id TEXT PRIMARY KEY,

        market_type TEXT NOT NULL,

        signal_created_at_utc TEXT NOT NULL,

        signal_batch_id TEXT NOT NULL,

        odds_api_event_id TEXT NOT NULL,

        commence_time_utc TEXT,

        hours_until_kickoff REAL,

        away_team TEXT,

        home_team TEXT,

        target_book TEXT,

        target_book_title TEXT,

        is_execution_book INTEGER,

        signal_side TEXT,

        signal_team TEXT,

        offered_line REAL,

        offered_price REAL,

        offered_break_even_probability REAL,

        target_fair_probability REAL,

        consensus_fair_probability REAL,

        consensus_home_value REAL,

        consensus_side_value REAL,

        consensus_books INTEGER,

        initial_disagreement REAL,

        research_rule TEXT,

        status TEXT
    )
    """
)


connection.commit()


# ============================================================
# LATEST PREGAME SNAPSHOT
# ============================================================

latest_capture = connection.execute(
    """
    SELECT MAX(captured_at_utc)

    FROM execution_market_snapshots

    WHERE is_pregame = 1
    """
).fetchone()[0]


if latest_capture is None:

    connection.close()

    raise RuntimeError(
        "No pregame market snapshots found."
    )


latest_batch = connection.execute(
    """
    SELECT batch_id

    FROM execution_market_snapshots

    WHERE
        is_pregame = 1
        AND captured_at_utc = ?

    LIMIT 1
    """,

    (
        latest_capture,
    )
).fetchone()


if latest_batch is None:

    connection.close()

    raise RuntimeError(
        "Could not identify latest snapshot batch."
    )


latest_batch = latest_batch[0]


data = pd.read_sql_query(
    """
    SELECT *

    FROM execution_market_snapshots

    WHERE
        is_pregame = 1
        AND captured_at_utc = ?

    ORDER BY
        odds_api_event_id,
        bookmaker_key
    """,

    connection,

    params=(
        latest_capture,
    )
)


if len(data) == 0:

    connection.close()

    raise RuntimeError(
        "Latest pregame snapshot contains no rows."
    )


print(
    f"Latest capture:    {latest_capture}"
)

print(
    f"Latest batch:      {latest_batch}"
)

print(
    f"Rows:              {len(data):,}"
)

print(
    f"Unique games:      "
    f"{data['odds_api_event_id'].nunique():,}"
)

print(
    f"Books represented: "
    f"{data['bookmaker_key'].nunique():,}"
)


# ============================================================
# NUMERIC CLEANING
# ============================================================

numeric_columns = [

    "hours_until_kickoff",

    "home_moneyline",
    "away_moneyline",

    "home_spread",
    "away_spread",

    "home_spread_price",
    "away_spread_price"
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


# ============================================================
# DE-VIG MONEYLINES
# ============================================================

data[
    "fair_home_probability"
] = data.apply(

    lambda row:

        fair_home_probability(

            row[
                "home_moneyline"
            ],

            row[
                "away_moneyline"
            ]
        ),

    axis=1
)


# ============================================================
# BUILD LEAVE-ONE-BOOK-OUT CONSENSUS
#
# If FanDuel is being evaluated, FanDuel is excluded from
# the consensus used to judge FanDuel.
# ============================================================

candidate_rows = []


for event_id, group in data.groupby(
    "odds_api_event_id"
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


        spread_values = (

            reference[
                "home_spread"
            ]

            .dropna()
        )


        moneyline_values = (

            reference[
                "fair_home_probability"
            ]

            .dropna()
        )


        spread_consensus = np.nan

        moneyline_consensus = np.nan


        if len(spread_values) > 0:

            spread_consensus = float(

                spread_values.median()
            )


        if len(moneyline_values) > 0:

            moneyline_consensus = float(

                moneyline_values.median()
            )


        spread_deviation = np.nan

        moneyline_deviation = np.nan


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
        ):

            spread_deviation = (

                float(
                    target[
                        "home_spread"
                    ]
                )

                -

                spread_consensus
            )


        if (
            pd.notna(
                target[
                    "fair_home_probability"
                ]
            )

            and

            pd.notna(
                moneyline_consensus
            )
        ):

            moneyline_deviation = (

                float(
                    target[
                        "fair_home_probability"
                    ]
                )

                -

                moneyline_consensus
            )


        row = target.to_dict()


        row[
            "spread_consensus"
        ] = spread_consensus


        row[
            "spread_consensus_books"
        ] = len(
            spread_values
        )


        row[
            "spread_deviation"
        ] = spread_deviation


        row[
            "spread_abs_deviation"
        ] = (

            abs(
                spread_deviation
            )

            if pd.notna(
                spread_deviation
            )

            else np.nan
        )


        row[
            "moneyline_consensus"
        ] = moneyline_consensus


        row[
            "moneyline_consensus_books"
        ] = len(
            moneyline_values
        )


        row[
            "moneyline_deviation"
        ] = moneyline_deviation


        row[
            "moneyline_abs_deviation_pp"
        ] = (

            abs(
                moneyline_deviation
            )
            *
            100.0

            if pd.notna(
                moneyline_deviation
            )

            else np.nan
        )


        candidate_rows.append(
            row
        )


candidates = pd.DataFrame(
    candidate_rows
)


# ============================================================
# EXISTING SIGNALS
#
# These are immutable for this experiment.
# ============================================================

existing_signal_ids = set(

    row[0]

    for row in connection.execute(
        """
        SELECT signal_id

        FROM prospective_market_signals_v1
        """
    ).fetchall()
)


# ============================================================
# SPREAD SIGNALS
#
# Require:
#
# - 3+ OTHER books
# - >= 1.0 point difference
# - spread price must exist
# - choose single largest outlier per game
# ============================================================

spread_candidates = candidates[

    (
        candidates[
            "spread_consensus_books"
        ]
        >=
        MINIMUM_OTHER_BOOKS
    )

    &

    (
        candidates[
            "spread_abs_deviation"
        ]
        >=
        SPREAD_SIGNAL_THRESHOLD
    )

    &

    candidates[
        "home_spread"
    ].notna()

    &

    candidates[
        "away_spread"
    ].notna()

].copy()


# Deterministic tie breaking:
#
# biggest disagreement first,
# then bookmaker name alphabetically.
spread_candidates = (

    spread_candidates

    .sort_values(

        [
            "odds_api_event_id",
            "spread_abs_deviation",
            "bookmaker_key"
        ],

        ascending=[
            True,
            False,
            True
        ]
    )

    .drop_duplicates(

        subset=[
            "odds_api_event_id"
        ],

        keep="first"
    )
)


new_signal_rows = []


for _, row in spread_candidates.iterrows():

    signal_id = (

        "spread_"

        +

        str(
            row[
                "odds_api_event_id"
            ]
        )
    )


    if signal_id in existing_signal_ids:

        continue


    # --------------------------------------------------------
    # Decide which SIDE receives the favorable number.
    #
    # Target home spread > consensus:
    #
    #     target +3.5
    #     market +2.5
    #
    # HOME has the better number.
    #
    #
    # Target home spread < consensus:
    #
    #     target -4.5
    #     market -3.5
    #
    # AWAY receives +4.5 instead of +3.5.
    # --------------------------------------------------------

    if row[
        "spread_deviation"
    ] > 0:

        signal_side = "HOME"

        signal_team = row[
            "home_team"
        ]

        offered_line = row[
            "home_spread"
        ]

        offered_price = row[
            "home_spread_price"
        ]

        consensus_side_value = row[
            "spread_consensus"
        ]


    else:

        signal_side = "AWAY"

        signal_team = row[
            "away_team"
        ]

        offered_line = row[
            "away_spread"
        ]

        offered_price = row[
            "away_spread_price"
        ]

        consensus_side_value = (

            -1.0

            *

            row[
                "spread_consensus"
            ]
        )


    # Exact juice matters.
    #
    # If no spread price is available, do not freeze a
    # supposedly actionable research signal.
    if pd.isna(
        offered_price
    ):

        continue


    offered_break_even = (
        american_to_probability(
            offered_price
        )
    )


    is_execution_book = int(

        row[
            "bookmaker_key"
        ]
        ==
        EXECUTION_BOOK
    )


    status = (

        "EXECUTABLE"

        if is_execution_book

        else "RESEARCH_ONLY"
    )


    new_signal_rows.append(
        {
            "signal_id":
                signal_id,

            "market_type":
                "spread",

            "signal_created_at_utc":
                latest_capture,

            "signal_batch_id":
                latest_batch,

            "odds_api_event_id":
                row[
                    "odds_api_event_id"
                ],

            "commence_time_utc":
                row[
                    "commence_time_utc"
                ],

            "hours_until_kickoff":
                row[
                    "hours_until_kickoff"
                ],

            "away_team":
                row[
                    "away_team"
                ],

            "home_team":
                row[
                    "home_team"
                ],

            "target_book":
                row[
                    "bookmaker_key"
                ],

            "target_book_title":
                row[
                    "bookmaker_title"
                ],

            "is_execution_book":
                is_execution_book,

            "signal_side":
                signal_side,

            "signal_team":
                signal_team,

            "offered_line":
                offered_line,

            "offered_price":
                offered_price,

            "offered_break_even_probability":
                offered_break_even,

            "target_fair_probability":
                np.nan,

            "consensus_fair_probability":
                np.nan,

            "consensus_home_value":
                row[
                    "spread_consensus"
                ],

            "consensus_side_value":
                consensus_side_value,

            "consensus_books":
                row[
                    "spread_consensus_books"
                ],

            "initial_disagreement":
                row[
                    "spread_abs_deviation"
                ],

            "research_rule":
                (
                    "largest outlier per game; "
                    "3+ other books; "
                    ">=1.0 point spread disagreement; "
                    "first qualifying signal frozen"
                ),

            "status":
                status
        }
    )


# ============================================================
# MONEYLINE SIGNALS
#
# Require:
#
# - 3+ OTHER valid moneyline books
# - >= 1.0 percentage point fair-probability difference
# - exact offered moneyline must exist
# - choose single largest outlier per game
# ============================================================

moneyline_candidates = candidates[

    (
        candidates[
            "moneyline_consensus_books"
        ]
        >=
        MINIMUM_OTHER_BOOKS
    )

    &

    (
        candidates[
            "moneyline_abs_deviation_pp"
        ]
        >=
        MONEYLINE_SIGNAL_THRESHOLD_PP
    )

    &

    candidates[
        "fair_home_probability"
    ].notna()

].copy()


moneyline_candidates = (

    moneyline_candidates

    .sort_values(

        [
            "odds_api_event_id",
            "moneyline_abs_deviation_pp",
            "bookmaker_key"
        ],

        ascending=[
            True,
            False,
            True
        ]
    )

    .drop_duplicates(

        subset=[
            "odds_api_event_id"
        ],

        keep="first"
    )
)


for _, row in moneyline_candidates.iterrows():

    signal_id = (

        "moneyline_"

        +

        str(
            row[
                "odds_api_event_id"
            ]
        )
    )


    if signal_id in existing_signal_ids:

        continue


    # --------------------------------------------------------
    # If target HOME fair probability is BELOW consensus,
    # HOME is cheaper at target.
    #
    # If target HOME fair probability is ABOVE consensus,
    # AWAY is cheaper at target.
    # --------------------------------------------------------

    if row[
        "moneyline_deviation"
    ] < 0:

        signal_side = "HOME"

        signal_team = row[
            "home_team"
        ]

        offered_price = row[
            "home_moneyline"
        ]

        target_fair_probability = row[
            "fair_home_probability"
        ]

        consensus_fair_probability = row[
            "moneyline_consensus"
        ]


    else:

        signal_side = "AWAY"

        signal_team = row[
            "away_team"
        ]

        offered_price = row[
            "away_moneyline"
        ]

        target_fair_probability = (

            1.0

            -

            row[
                "fair_home_probability"
            ]
        )

        consensus_fair_probability = (

            1.0

            -

            row[
                "moneyline_consensus"
            ]
        )


    if pd.isna(
        offered_price
    ):

        continue


    offered_break_even = (
        american_to_probability(
            offered_price
        )
    )


    is_execution_book = int(

        row[
            "bookmaker_key"
        ]
        ==
        EXECUTION_BOOK
    )


    status = (

        "EXECUTABLE"

        if is_execution_book

        else "RESEARCH_ONLY"
    )


    new_signal_rows.append(
        {
            "signal_id":
                signal_id,

            "market_type":
                "moneyline",

            "signal_created_at_utc":
                latest_capture,

            "signal_batch_id":
                latest_batch,

            "odds_api_event_id":
                row[
                    "odds_api_event_id"
                ],

            "commence_time_utc":
                row[
                    "commence_time_utc"
                ],

            "hours_until_kickoff":
                row[
                    "hours_until_kickoff"
                ],

            "away_team":
                row[
                    "away_team"
                ],

            "home_team":
                row[
                    "home_team"
                ],

            "target_book":
                row[
                    "bookmaker_key"
                ],

            "target_book_title":
                row[
                    "bookmaker_title"
                ],

            "is_execution_book":
                is_execution_book,

            "signal_side":
                signal_side,

            "signal_team":
                signal_team,

            "offered_line":
                np.nan,

            "offered_price":
                offered_price,

            "offered_break_even_probability":
                offered_break_even,

            "target_fair_probability":
                target_fair_probability,

            "consensus_fair_probability":
                consensus_fair_probability,

            "consensus_home_value":
                row[
                    "moneyline_consensus"
                ],

            "consensus_side_value":
                consensus_fair_probability,

            "consensus_books":
                row[
                    "moneyline_consensus_books"
                ],

            "initial_disagreement":
                row[
                    "moneyline_abs_deviation_pp"
                ],

            "research_rule":
                (
                    "largest outlier per game; "
                    "3+ other books; "
                    ">=1.0 pp fair-probability disagreement; "
                    "first qualifying signal frozen"
                ),

            "status":
                status
        }
    )


# ============================================================
# INSERT NEW SIGNALS
# ============================================================

signal_columns = [

    "signal_id",
    "market_type",
    "signal_created_at_utc",
    "signal_batch_id",
    "odds_api_event_id",
    "commence_time_utc",
    "hours_until_kickoff",
    "away_team",
    "home_team",
    "target_book",
    "target_book_title",
    "is_execution_book",
    "signal_side",
    "signal_team",
    "offered_line",
    "offered_price",
    "offered_break_even_probability",
    "target_fair_probability",
    "consensus_fair_probability",
    "consensus_home_value",
    "consensus_side_value",
    "consensus_books",
    "initial_disagreement",
    "research_rule",
    "status"
]


if len(
    new_signal_rows
) > 0:

    placeholders = ",".join(
        [
            "?"
            for _ in signal_columns
        ]
    )


    insert_values = []


    for row in new_signal_rows:

        values = []


        for column in signal_columns:

            value = row[
                column
            ]


            if pd.isna(
                value
            ):

                value = None


            values.append(
                value
            )


        insert_values.append(
            tuple(
                values
            )
        )


    connection.executemany(
        f"""
        INSERT OR IGNORE INTO
        prospective_market_signals_v1 (

            {",".join(signal_columns)}

        )

        VALUES (
            {placeholders}
        )
        """,

        insert_values
    )


    connection.commit()


# ============================================================
# READ PERMANENT LEDGER
# ============================================================

ledger = pd.read_sql_query(
    """
    SELECT *

    FROM prospective_market_signals_v1

    ORDER BY
        signal_created_at_utc,
        odds_api_event_id,
        market_type
    """,

    connection
)


connection.close()


# ============================================================
# EXPORT CSV COPY
# ============================================================

os.makedirs(
    "data/processed",
    exist_ok=True
)


ledger.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# IDENTIFY WHAT WAS ACTUALLY NEW
# ============================================================

new_ids = {

    row[
        "signal_id"
    ]

    for row in new_signal_rows

    if (
        row[
            "signal_id"
        ]
        not in existing_signal_ids
    )
}


new_signals = ledger[

    ledger[
        "signal_id"
    ].isin(
        new_ids
    )

].copy()


# ============================================================
# SUMMARY
# ============================================================

print()

print(
    "=" * 115
)

print(
    "SIGNALS FROZEN THIS RUN"
)

print(
    "=" * 115
)


new_spreads = new_signals[

    new_signals[
        "market_type"
    ]
    ==
    "spread"

]


new_moneylines = new_signals[

    new_signals[
        "market_type"
    ]
    ==
    "moneyline"

]


print(
    f"New spread signals:    "
    f"{len(new_spreads):,}"
)


print(
    f"New moneyline signals: "
    f"{len(new_moneylines):,}"
)


print(
    f"Total new signals:     "
    f"{len(new_signals):,}"
)


print(
    f"Permanent ledger size: "
    f"{len(ledger):,}"
)


# ============================================================
# SHOW NEW SIGNALS
# ============================================================

if len(
    new_signals
) > 0:

    print()

    print(
        "NEWLY FROZEN SIGNALS"
    )

    print(
        "-" * 115
    )


    display_columns = [

        "market_type",
        "away_team",
        "home_team",
        "target_book",
        "signal_side",
        "signal_team",
        "offered_line",
        "offered_price",
        "consensus_side_value",
        "consensus_books",
        "initial_disagreement",
        "hours_until_kickoff",
        "status"
    ]


    print(

        new_signals[
            display_columns
        ]

        .sort_values(

            [
                "market_type",
                "initial_disagreement"
            ],

            ascending=[
                True,
                False
            ]
        )

        .to_string(
            index=False
        )
    )


# ============================================================
# EXECUTION STATUS
# ============================================================

execution_signals = new_signals[

    new_signals[
        "is_execution_book"
    ]
    ==
    1

].copy()


print()

print(
    "=" * 115
)

print(
    "EXECUTION STATUS"
)

print(
    "=" * 115
)


if len(
    execution_signals
) == 0:

    print(
        "No Hard Rock FL NCAAF execution signals."
    )

    print(
        "All signals frozen today are research-only."
    )


else:

    print(
        f"Hard Rock execution signals: "
        f"{len(execution_signals):,}"
    )


# ============================================================
# RULE CONFIRMATION
# ============================================================

print()

print(
    "=" * 115
)

print(
    "FROZEN RESEARCH RULE"
)

print(
    "=" * 115
)


print(
    "Spread:"
)

print(
    "  3+ other books"
)

print(
    "  >= 1.0 point disagreement"
)

print(
    "  largest outlier per game"
)

print(
    "  first qualifying signal only"
)


print()

print(
    "Moneyline:"
)

print(
    "  3+ other books"
)

print(
    "  >= 1.0 percentage-point fair-probability disagreement"
)

print(
    "  largest outlier per game"
)

print(
    "  first qualifying signal only"
)


print()

print(
    "Signals are never overwritten."
)

print(
    "Future evaluation will use the final "
    "high-quality pregame consensus."
)


print()

print(
    "=" * 115
)

print(
    "FREEZE COMPLETE"
)

print(
    "=" * 115
)


print(
    OUTPUT_FILE
)