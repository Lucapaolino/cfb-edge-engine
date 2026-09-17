import json
import os
import sqlite3

from datetime import datetime, timezone

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

DATABASE_FILE = "data/cfb.db"

OUTPUT_DIRECTORY = "data/processed"

JSON_OUTPUT = os.path.join(
    OUTPUT_DIRECTORY,
    "game_day_board_v1.json"
)

CSV_OUTPUT = os.path.join(
    OUTPUT_DIRECTORY,
    "game_day_board_v1.csv"
)


# ============================================================
# HELPERS
# ============================================================

def parse_utc_datetime(value):

    if pd.isna(value):

        return None


    try:

        parsed = datetime.fromisoformat(
            str(value).replace(
                "Z",
                "+00:00"
            )
        )


        if parsed.tzinfo is None:

            parsed = parsed.replace(
                tzinfo=timezone.utc
            )


        return parsed.astimezone(
            timezone.utc
        )


    except ValueError:

        return None


def format_price(value):

    if pd.isna(value):

        return None


    value = int(
        round(
            float(value)
        )
    )


    if value > 0:

        return f"+{value}"


    return str(
        value
    )


def format_spread(value):

    if pd.isna(value):

        return None


    value = float(
        value
    )


    if value > 0:

        return f"+{value:g}"


    return f"{value:g}"


# ============================================================
# START
# ============================================================

print(
    "BUILD GAME DAY BOARD — V1"
)

print(
    "=" * 110
)


os.makedirs(
    OUTPUT_DIRECTORY,
    exist_ok=True
)


connection = sqlite3.connect(
    DATABASE_FILE
)


# ============================================================
# VERIFY TABLE EXISTS
# ============================================================

table_exists = connection.execute(
    """
    SELECT name

    FROM sqlite_master

    WHERE
        type = 'table'
        AND name = 'prospective_market_signals_v1'
    """
).fetchone()


if table_exists is None:

    connection.close()

    raise RuntimeError(
        "prospective_market_signals_v1 table does not exist."
    )


# ============================================================
# LOAD SIGNAL LEDGER
# ============================================================

signals = pd.read_sql_query(
    """
    SELECT *

    FROM prospective_market_signals_v1

    ORDER BY
        commence_time_utc,
        market_type,
        signal_id
    """,

    connection
)


connection.close()


print(
    f"Frozen signals loaded: "
    f"{len(signals):,}"
)


# ============================================================
# EMPTY BOARD
# ============================================================

if len(
    signals
) == 0:

    empty_payload = {

        "generated_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "mode":
            "RESEARCH",

        "summary": {

            "active_signals": 0,

            "spread_signals": 0,

            "moneyline_signals": 0,

            "execution_signals": 0
        },

        "signals":
            []
    }


    with open(
        JSON_OUTPUT,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            empty_payload,
            file,
            indent=2
        )


    pd.DataFrame().to_csv(
        CSV_OUTPUT,
        index=False
    )


    print(
        "No frozen signals."
    )


    raise SystemExit(
        0
    )


# ============================================================
# NUMERIC CLEANING
# ============================================================

numeric_columns = [

    "hours_until_kickoff",

    "offered_line",

    "offered_price",

    "offered_break_even_probability",

    "target_fair_probability",

    "consensus_fair_probability",

    "consensus_home_value",

    "consensus_side_value",

    "consensus_books",

    "initial_disagreement",

    "is_execution_book"
]


for column in numeric_columns:

    if column in signals.columns:

        signals[
            column
        ] = pd.to_numeric(

            signals[
                column
            ],

            errors="coerce"
        )


# ============================================================
# CURRENT TIME / ACTIVE GAMES
# ============================================================

now_utc = datetime.now(
    timezone.utc
)


signals[
    "kickoff_datetime"
] = signals[
    "commence_time_utc"
].apply(
    parse_utc_datetime
)


def calculate_hours_until_kickoff(
    kickoff
):

    if kickoff is None:

        return np.nan


    return (

        kickoff
        -
        now_utc

    ).total_seconds() / 3600.0


signals[
    "current_hours_until_kickoff"
] = signals[
    "kickoff_datetime"
].apply(
    calculate_hours_until_kickoff
)


# ============================================================
# ONLY SHOW GAMES THAT HAVE NOT STARTED
# ============================================================

active = signals[

    signals[
        "current_hours_until_kickoff"
    ].notna()

    &

    (
        signals[
            "current_hours_until_kickoff"
        ]
        >
        0
    )

].copy()


print(
    f"Active pregame signals: "
    f"{len(active):,}"
)


# ============================================================
# STRENGTH PLACEHOLDERS
#
# VERY IMPORTANT:
#
# We are NOT inventing a strength formula yet.
#
# These fields are intentionally empty until historical +
# prospective validation tells us how strength should behave.
# ============================================================

active[
    "strength_score"
] = np.nan


active[
    "strength_label"
] = "PENDING VALIDATION"


# ============================================================
# DISPLAY VALUES
# ============================================================

active[
    "display_price"
] = active[
    "offered_price"
].apply(
    format_price
)


active[
    "display_line"
] = active[
    "offered_line"
].apply(
    format_spread
)


def create_pick_text(
    row
):

    if row[
        "market_type"
    ] == "spread":

        line = row[
            "display_line"
        ]


        price = row[
            "display_price"
        ]


        return (
            f"{row['signal_team']} "
            f"{line} "
            f"({price})"
        )


    price = row[
        "display_price"
    ]


    return (
        f"{row['signal_team']} ML "
        f"({price})"
    )


active[
    "display_pick"
] = active.apply(
    create_pick_text,
    axis=1
)


# ============================================================
# HUMAN-READABLE MARKET DISAGREEMENT
#
# This is NOT Strength.
#
# This simply describes how far the target sportsbook was
# from consensus when the signal fired.
# ============================================================

def disagreement_text(
    row
):

    value = row[
        "initial_disagreement"
    ]


    if pd.isna(value):

        return None


    if row[
        "market_type"
    ] == "spread":

        return (
            f"{value:.2f} pts"
        )


    return (
        f"{value:.2f} pp"
    )


active[
    "display_disagreement"
] = active.apply(
    disagreement_text,
    axis=1
)


# ============================================================
# SORT
#
# Research-only for now.
#
# Larger disagreement first within each market.
# ============================================================

active = active.sort_values(

    [
        "market_type",
        "initial_disagreement",
        "current_hours_until_kickoff"
    ],

    ascending=[
        True,
        False,
        True
    ]
)


# ============================================================
# BUILD JSON RECORDS
# ============================================================

board_records = []


for _, row in active.iterrows():

    board_records.append(
        {
            "signal_id":
                row[
                    "signal_id"
                ],

            "market_type":
                row[
                    "market_type"
                ],

            "away_team":
                row[
                    "away_team"
                ],

            "home_team":
                row[
                    "home_team"
                ],

            "game":
                (
                    f"{row['away_team']} "
                    f"@ "
                    f"{row['home_team']}"
                ),

            "kickoff_utc":
                row[
                    "commence_time_utc"
                ],

            "hours_until_kickoff":
                round(
                    float(
                        row[
                            "current_hours_until_kickoff"
                        ]
                    ),
                    2
                ),

            "target_book":
                row[
                    "target_book"
                ],

            "signal_side":
                row[
                    "signal_side"
                ],

            "signal_team":
                row[
                    "signal_team"
                ],

            "pick":
                row[
                    "display_pick"
                ],

            "offered_line":
                (
                    None

                    if pd.isna(
                        row[
                            "offered_line"
                        ]
                    )

                    else float(
                        row[
                            "offered_line"
                        ]
                    )
                ),

            "offered_price":
                (
                    None

                    if pd.isna(
                        row[
                            "offered_price"
                        ]
                    )

                    else int(
                        round(
                            float(
                                row[
                                    "offered_price"
                                ]
                            )
                        )
                    )
                ),

            "consensus_books":
                int(
                    row[
                        "consensus_books"
                    ]
                ),

            "initial_disagreement":
                float(
                    row[
                        "initial_disagreement"
                    ]
                ),

            "disagreement_display":
                row[
                    "display_disagreement"
                ],

            "strength_score":
                None,

            "strength_label":
                "PENDING VALIDATION",

            "execution_status":
                row[
                    "status"
                ],

            "is_execution_book":
                bool(
                    row[
                        "is_execution_book"
                    ]
                )
        }
    )


# ============================================================
# SUMMARY
# ============================================================

spread_count = int(

    (
        active[
            "market_type"
        ]
        ==
        "spread"
    ).sum()
)


moneyline_count = int(

    (
        active[
            "market_type"
        ]
        ==
        "moneyline"
    ).sum()
)


execution_count = int(

    (
        active[
            "is_execution_book"
        ]
        ==
        1
    ).sum()
)


payload = {

    "generated_at_utc":
        now_utc.replace(
            microsecond=0
        ).isoformat(),

    "mode":
        "RESEARCH",

    "summary": {

        "active_signals":
            len(
                active
            ),

        "spread_signals":
            spread_count,

        "moneyline_signals":
            moneyline_count,

        "execution_signals":
            execution_count,

        "strength_model":
            "NOT YET VALIDATED"
    },

    "signals":
        board_records
}


# ============================================================
# SAVE JSON
# ============================================================

with open(
    JSON_OUTPUT,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        payload,
        file,
        indent=2
    )


# ============================================================
# SAVE CSV
# ============================================================

csv_columns = [

    "signal_id",

    "market_type",

    "away_team",

    "home_team",

    "signal_team",

    "display_pick",

    "target_book",

    "offered_line",

    "offered_price",

    "consensus_books",

    "initial_disagreement",

    "display_disagreement",

    "current_hours_until_kickoff",

    "strength_score",

    "strength_label",

    "status"
]


active[
    csv_columns
].to_csv(
    CSV_OUTPUT,
    index=False
)


# ============================================================
# OUTPUT
# ============================================================

print()

print(
    "=" * 110
)

print(
    "GAME DAY BOARD"
)

print(
    "=" * 110
)


print(
    f"Active signals:      "
    f"{len(active):,}"
)


print(
    f"Spread signals:      "
    f"{spread_count:,}"
)


print(
    f"Moneyline signals:   "
    f"{moneyline_count:,}"
)


print(
    f"Executable signals:  "
    f"{execution_count:,}"
)


print()

print(
    "Strength model:"
)

print(
    "  PENDING VALIDATION"
)


print()

print(
    f"JSON: {JSON_OUTPUT}"
)


print(
    f"CSV:  {CSV_OUTPUT}"
)


print()

print(
    "=" * 110
)

print(
    "BUILD COMPLETE"
)

print(
    "=" * 110
)