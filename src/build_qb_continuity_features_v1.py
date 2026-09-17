import os
import sqlite3

import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

DATABASE_FILE = "data/cfb.db"

INPUT_FILE = (
    "data/processed/"
    "features_v3.csv"
)

OUTPUT_FILE = (
    "data/processed/"
    "features_v4.csv"
)


SEASONS = [
    2021,
    2022,
    2023,
    2024,
    2025
]


# ============================================================
# START
# ============================================================

print(
    "BUILD QB CONTINUITY FEATURES — V1"
)

print(
    "=" * 110
)


features = pd.read_csv(
    INPUT_FILE
)


print(
    f"Feature rows loaded:    "
    f"{len(features):,}"
)


print(
    f"Feature columns loaded: "
    f"{len(features.columns):,}"
)


# ============================================================
# REQUIRED MODEL COLUMNS
# ============================================================

required_columns = [
    "game_id",
    "season",
    "week",
    "home_team",
    "away_team"
]


missing = [
    column
    for column in required_columns
    if column not in features.columns
]


if missing:

    raise RuntimeError(
        "Missing required feature columns: "
        +
        ", ".join(missing)
    )


# ============================================================
# LOAD HISTORICAL PASSING C/ATT
#
# We use ALL FBS-involving games stored in the database,
# including prior FBS-v-FCS games.
#
# For personnel continuity, playing an FCS opponent still
# tells us who was taking the snaps.
# ============================================================

connection = sqlite3.connect(
    DATABASE_FILE
)


passing = pd.read_sql_query(

    """
    SELECT
        p.game_id,
        g.season,
        g.week,
        p.team,
        p.athlete_id,
        p.athlete_name,
        p.stat_value AS completion_attempt_value

    FROM player_game_stat_values p

    JOIN games g
        ON g.game_id = p.game_id

    WHERE p.category = 'passing'

    AND p.stat_type = 'C/ATT'

    AND g.season BETWEEN 2021 AND 2025

    ORDER BY
        g.season,
        g.week,
        p.game_id,
        p.team
    """,

    connection
)


connection.close()


print()

print(
    f"C/ATT passer rows loaded: "
    f"{len(passing):,}"
)


# ============================================================
# CLEAN IDS
# ============================================================

passing[
    "game_id"
] = pd.to_numeric(

    passing[
        "game_id"
    ],

    errors="coerce"
)


passing[
    "season"
] = pd.to_numeric(

    passing[
        "season"
    ],

    errors="coerce"
)


passing[
    "week"
] = pd.to_numeric(

    passing[
        "week"
    ],

    errors="coerce"
)


passing[
    "athlete_id"
] = passing[
    "athlete_id"
].astype(
    str
)


# ============================================================
# PARSE C/ATT
#
# Example:
#
# 19/31
#
# means:
#
# 19 completions
# 31 attempts
# ============================================================

def parse_completions_attempts(
    value
):

    if pd.isna(
        value
    ):

        return (
            np.nan,
            np.nan
        )


    value = str(
        value
    ).strip()


    if "/" not in value:

        return (
            np.nan,
            np.nan
        )


    pieces = value.split(
        "/"
    )


    if len(
        pieces
    ) != 2:

        return (
            np.nan,
            np.nan
        )


    try:

        completions = float(
            pieces[0]
        )


        attempts = float(
            pieces[1]
        )


    except ValueError:

        return (
            np.nan,
            np.nan
        )


    return (
        completions,
        attempts
    )


parsed = passing[
    "completion_attempt_value"
].apply(
    parse_completions_attempts
)


passing[
    "completions"
] = [
    value[0]
    for value in parsed
]


passing[
    "attempts"
] = [
    value[1]
    for value in parsed
]


passing = passing.dropna(
    subset=[
        "game_id",
        "season",
        "week",
        "team",
        "athlete_id",
        "attempts"
    ]
).copy()


passing[
    "game_id"
] = passing[
    "game_id"
].astype(
    int
)


passing[
    "season"
] = passing[
    "season"
].astype(
    int
)


passing[
    "week"
] = passing[
    "week"
].astype(
    int
)


# ============================================================
# BASIC PARSING AUDIT
# ============================================================

print()

print(
    "PASSING ATTEMPT AUDIT"
)

print(
    "-" * 110
)


print(

    f"Parsed passer rows: "
    f"{len(passing):,}"
)


print(

    f"Total pass attempts: "
    f"{passing['attempts'].sum():,.0f}"
)


print(

    f"Rows with zero attempts: "
    f"{(passing['attempts'] == 0).sum():,}"
)


# ============================================================
# PRIMARY PASSER BY TEAM/GAME
#
# Primary passer =
# player with the most pass attempts for that team/game.
#
# This does NOT necessarily mean official "starter."
#
# It means the QB/player who handled the largest passing
# workload in the completed game.
# ============================================================

team_game_rows = []


grouped = passing.groupby(
    [
        "season",
        "week",
        "game_id",
        "team"
    ],
    sort=False
)


for (
    season,
    week,
    game_id,
    team
), group in grouped:

    group = group.copy()


    team_attempts = float(
        group[
            "attempts"
        ].sum()
    )


    # --------------------------------------------------------
    # Deterministic tie handling:
    #
    # 1. Most attempts
    # 2. Most completions
    # 3. Athlete ID
    # --------------------------------------------------------

    group = group.sort_values(

        [
            "attempts",
            "completions",
            "athlete_id"
        ],

        ascending=[
            False,
            False,
            True
        ]
    )


    primary = group.iloc[
        0
    ]


    primary_attempts = float(
        primary[
            "attempts"
        ]
    )


    primary_share = (

        primary_attempts
        /
        team_attempts

        if team_attempts > 0

        else
        np.nan
    )


    team_game_rows.append(
        {
            "season":
                int(
                    season
                ),

            "week":
                int(
                    week
                ),

            "game_id":
                int(
                    game_id
                ),

            "team":
                team,

            "primary_athlete_id":
                str(
                    primary[
                        "athlete_id"
                    ]
                ),

            "primary_athlete_name":
                primary[
                    "athlete_name"
                ],

            "primary_attempts":
                primary_attempts,

            "team_attempts":
                team_attempts,

            "primary_attempt_share":
                primary_share,

            "number_of_passers":
                len(
                    group
                )
        }
    )


team_games = pd.DataFrame(
    team_game_rows
)


print()

print(
    f"Team-game passing records: "
    f"{len(team_games):,}"
)


print(

    f"Unique games represented: "
    f"{team_games['game_id'].nunique():,}"
)


# ============================================================
# FEATURE FUNCTION
#
# Input:
# all COMPLETED prior games for one team.
#
# Output:
# what we would have known entering the next game.
# ============================================================

def summarize_qb_history(
    history
):

    # --------------------------------------------------------
    # No prior passer history.
    # --------------------------------------------------------

    if len(
        history
    ) == 0:

        return {
            "history_games":
                0.0,

            "distinct_primary_passers":
                np.nan,

            "primary_streak":
                np.nan,

            "last_primary_start_share":
                np.nan,

            "last_primary_attempt_share":
                np.nan,

            "last_game_primary_attempt_share":
                np.nan,

            "top_passer_attempt_share":
                np.nan,

            "qb_change_rate":
                np.nan,

            "same_primary_last2":
                np.nan,

            "continuity_available":
                0.0
        }


    history = history.sort_values(
        [
            "week",
            "game_id"
        ]
    ).copy()


    history_games = len(
        history
    )


    primary_ids = history[
        "primary_athlete_id"
    ].astype(
        str
    ).tolist()


    last_primary = primary_ids[
        -1
    ]


    # ========================================================
    # DISTINCT PRIMARY PASSERS
    # ========================================================

    distinct_primary_passers = len(
        set(
            primary_ids
        )
    )


    # ========================================================
    # CURRENT STREAK
    #
    # Number of consecutive prior games ending with the same
    # primary passer.
    #
    # Example:
    #
    # QB A
    # QB B
    # QB B
    # QB B
    #
    # streak = 3
    # ========================================================

    primary_streak = 0


    for athlete_id in reversed(
        primary_ids
    ):

        if athlete_id == last_primary:

            primary_streak += 1

        else:

            break


    # ========================================================
    # START SHARE
    #
    # Fraction of prior team games in which the most recent
    # primary passer was also primary passer.
    # ========================================================

    last_primary_starts = sum(

        athlete_id
        ==
        last_primary

        for athlete_id
        in primary_ids
    )


    last_primary_start_share = (

        last_primary_starts
        /
        history_games
    )


    # ========================================================
    # ATTEMPT SHARE
    #
    # Fraction of ALL prior team pass attempts belonging to
    # the most recent primary passer.
    # ========================================================

    total_team_attempts = float(
        history[
            "team_attempts"
        ].sum()
    )


    last_primary_attempts = float(

        history.loc[
            history[
                "primary_athlete_id"
            ].astype(
                str
            )
            ==
            last_primary,

            "primary_attempts"
        ].sum()
    )


    last_primary_attempt_share = (

        last_primary_attempts
        /
        total_team_attempts

        if total_team_attempts > 0

        else
        np.nan
    )


    # ========================================================
    # LAST GAME CONTROL
    # ========================================================

    last_game_primary_attempt_share = float(

        history.iloc[
            -1
        ][
            "primary_attempt_share"
        ]
    )


    # ========================================================
    # SEASON PASSER CONCENTRATION
    #
    # Which passer has accounted for the greatest cumulative
    # number of primary-passer attempts?
    # ========================================================

    passer_attempt_totals = (

        history

        .groupby(
            "primary_athlete_id"
        )[
            "primary_attempts"
        ]

        .sum()
    )


    top_passer_attempts = float(
        passer_attempt_totals.max()
    )


    top_passer_attempt_share = (

        top_passer_attempts
        /
        total_team_attempts

        if total_team_attempts > 0

        else
        np.nan
    )


    # ========================================================
    # QB CHANGE RATE
    #
    # Example:
    #
    # A A B B C
    #
    # transitions:
    #
    # A -> A = no change
    # A -> B = change
    # B -> B = no change
    # B -> C = change
    #
    # change rate = 2 / 4 = 50%
    # ========================================================

    if history_games >= 2:

        changes = sum(

            primary_ids[index]
            !=
            primary_ids[index - 1]

            for index
            in range(
                1,
                history_games
            )
        )


        qb_change_rate = (

            changes
            /
            (
                history_games
                -
                1
            )
        )


        same_primary_last2 = float(

            primary_ids[
                -1
            ]

            ==
            primary_ids[
                -2
            ]
        )


        continuity_available = 1.0


    else:

        qb_change_rate = np.nan

        same_primary_last2 = np.nan

        continuity_available = 0.0


    return {
        "history_games":
            float(
                history_games
            ),

        "distinct_primary_passers":
            float(
                distinct_primary_passers
            ),

        "primary_streak":
            float(
                primary_streak
            ),

        "last_primary_start_share":
            float(
                last_primary_start_share
            ),

        "last_primary_attempt_share":
            float(
                last_primary_attempt_share
            ),

        "last_game_primary_attempt_share":
            float(
                last_game_primary_attempt_share
            ),

        "top_passer_attempt_share":
            float(
                top_passer_attempt_share
            ),

        "qb_change_rate":
            float(
                qb_change_rate
            )
            if not pd.isna(
                qb_change_rate
            )
            else
            np.nan,

        "same_primary_last2":
            float(
                same_primary_last2
            )
            if not pd.isna(
                same_primary_last2
            )
            else
            np.nan,

        "continuity_available":
            float(
                continuity_available
            )
    }


# ============================================================
# QB FEATURE NAMES
# ============================================================

TEAM_FEATURE_NAMES = [

    "history_games",

    "distinct_primary_passers",

    "primary_streak",

    "last_primary_start_share",

    "last_primary_attempt_share",

    "last_game_primary_attempt_share",

    "top_passer_attempt_share",

    "qb_change_rate",

    "same_primary_last2",

    "continuity_available"
]


# ============================================================
# OUTPUT STORAGE
# ============================================================

home_feature_values = {

    feature:
        np.full(
            len(
                features
            ),
            np.nan
        )

    for feature
    in TEAM_FEATURE_NAMES
}


away_feature_values = {

    feature:
        np.full(
            len(
                features
            ),
            np.nan
        )

    for feature
    in TEAM_FEATURE_NAMES
}


# ============================================================
# BUILD PREGAME FEATURES
#
# CRITICAL LEAKAGE RULE:
#
# only historical passer records with
#
#     historical_week < current_week
#
# are allowed.
#
# No current-week passer information is used.
# ============================================================

print()

print(
    "=" * 110
)

print(
    "BUILDING PREGAME QB FEATURES"
)

print(
    "=" * 110
)


for season in SEASONS:

    season_feature_rows = features[
        features[
            "season"
        ]
        ==
        season
    ]


    if len(
        season_feature_rows
    ) == 0:

        continue


    season_team_games = team_games[
        team_games[
            "season"
        ]
        ==
        season
    ].copy()


    print()

    print(
        f"Season {season}"
    )

    print(
        "-" * 70
    )


    rows_with_any_history = 0

    rows_with_both_continuity = 0


    for index, game in season_feature_rows.iterrows():

        current_week = int(
            game[
                "week"
            ]
        )


        home_team = game[
            "home_team"
        ]


        away_team = game[
            "away_team"
        ]


        # ----------------------------------------------------
        # STRICTLY PRIOR WEEKS
        # ----------------------------------------------------

        home_history = season_team_games[
            (
                season_team_games[
                    "team"
                ]
                ==
                home_team
            )
            &
            (
                season_team_games[
                    "week"
                ]
                <
                current_week
            )
        ].copy()


        away_history = season_team_games[
            (
                season_team_games[
                    "team"
                ]
                ==
                away_team
            )
            &
            (
                season_team_games[
                    "week"
                ]
                <
                current_week
            )
        ].copy()


        home_summary = summarize_qb_history(
            home_history
        )


        away_summary = summarize_qb_history(
            away_history
        )


        for feature_name in TEAM_FEATURE_NAMES:

            home_feature_values[
                feature_name
            ][
                index
            ] = home_summary[
                feature_name
            ]


            away_feature_values[
                feature_name
            ][
                index
            ] = away_summary[
                feature_name
            ]


        if (
            home_summary[
                "history_games"
            ]
            >
            0
            or
            away_summary[
                "history_games"
            ]
            >
            0
        ):

            rows_with_any_history += 1


        if (
            home_summary[
                "continuity_available"
            ]
            ==
            1
            and
            away_summary[
                "continuity_available"
            ]
            ==
            1
        ):

            rows_with_both_continuity += 1


    print(

        f"Games with some prior QB history: "
        f"{rows_with_any_history:,} / "
        f"{len(season_feature_rows):,}"
    )


    print(

        f"Games with continuity on BOTH teams: "
        f"{rows_with_both_continuity:,} / "
        f"{len(season_feature_rows):,}"
    )


# ============================================================
# ADD HOME / AWAY FEATURES ALL AT ONCE
#
# Using concat avoids the DataFrame fragmentation warning
# we saw in the previous builder.
# ============================================================

new_columns = {}


for feature_name in TEAM_FEATURE_NAMES:

    new_columns[
        f"qb_home_{feature_name}"
    ] = home_feature_values[
        feature_name
    ]


    new_columns[
        f"qb_away_{feature_name}"
    ] = away_feature_values[
        feature_name
    ]


# ============================================================
# MATCHUP DIFFERENCES
#
# Home minus away.
# ============================================================

DIFF_FEATURE_NAMES = [

    "history_games",

    "distinct_primary_passers",

    "primary_streak",

    "last_primary_start_share",

    "last_primary_attempt_share",

    "last_game_primary_attempt_share",

    "top_passer_attempt_share",

    "qb_change_rate",

    "same_primary_last2"
]


for feature_name in DIFF_FEATURE_NAMES:

    new_columns[
        f"qb_diff_{feature_name}"
    ] = (

        home_feature_values[
            feature_name
        ]

        -

        away_feature_values[
            feature_name
        ]
    )


qb_frame = pd.DataFrame(
    new_columns,
    index=features.index
)


features_v4 = pd.concat(

    [
        features,
        qb_frame
    ],

    axis=1
)


# ============================================================
# LEAKAGE AUDIT
#
# Week 1 MUST have no previous QB history.
# ============================================================

week_one = features_v4[
    features_v4[
        "week"
    ]
    ==
    1
]


week_one_home_history = (
    week_one[
        "qb_home_history_games"
    ]
    >
    0
).sum()


week_one_away_history = (
    week_one[
        "qb_away_history_games"
    ]
    >
    0
).sum()


print()

print(
    "=" * 110
)

print(
    "QB FEATURE LEAKAGE AUDIT"
)

print(
    "=" * 110
)


print(

    f"Week 1 rows with HOME prior QB history: "
    f"{week_one_home_history}"
)


print(

    f"Week 1 rows with AWAY prior QB history: "
    f"{week_one_away_history}"
)


if (
    week_one_home_history != 0
    or
    week_one_away_history != 0
):

    raise RuntimeError(
        "Leakage detected: Week 1 contains "
        "prior QB information."
    )


# ============================================================
# COVERAGE AUDIT
# ============================================================

audit_rows = []


for season in SEASONS:

    subset = features_v4[
        features_v4[
            "season"
        ]
        ==
        season
    ]


    if len(
        subset
    ) == 0:

        continue


    home_history = (

        subset[
            "qb_home_history_games"
        ]
        >
        0
    )


    away_history = (

        subset[
            "qb_away_history_games"
        ]
        >
        0
    )


    both_history = (
        home_history
        &
        away_history
    )


    both_continuity = (

        (
            subset[
                "qb_home_continuity_available"
            ]
            ==
            1
        )

        &

        (
            subset[
                "qb_away_continuity_available"
            ]
            ==
            1
        )
    )


    audit_rows.append(
        {
            "season":
                season,

            "games":
                len(
                    subset
                ),

            "both_teams_have_prior_qb_game":
                int(
                    both_history.sum()
                ),

            "both_teams_have_continuity":
                int(
                    both_continuity.sum()
                ),

            "prior_qb_coverage":
                both_history.mean(),

            "continuity_coverage":
                both_continuity.mean()
        }
    )


audit = pd.DataFrame(
    audit_rows
)


print()

print(
    "=" * 110
)

print(
    "QB FEATURE COVERAGE"
)

print(
    "=" * 110
)


print(

    audit.to_string(

        index=False,

        formatters={

            "prior_qb_coverage":
                lambda x:
                    f"{x:.2%}",

            "continuity_coverage":
                lambda x:
                    f"{x:.2%}"
        }
    )
)


# ============================================================
# FEATURE RANGE AUDIT
# ============================================================

print()

print(
    "=" * 110
)

print(
    "QB FEATURE RANGES"
)

print(
    "=" * 110
)


range_columns = [

    "qb_home_primary_streak",

    "qb_away_primary_streak",

    "qb_diff_primary_streak",

    "qb_home_last_primary_start_share",

    "qb_away_last_primary_start_share",

    "qb_diff_last_primary_start_share",

    "qb_home_last_primary_attempt_share",

    "qb_away_last_primary_attempt_share",

    "qb_diff_last_primary_attempt_share",

    "qb_home_qb_change_rate",

    "qb_away_qb_change_rate",

    "qb_diff_qb_change_rate"
]


for column in range_columns:

    valid = features_v4[
        column
    ].dropna()


    if len(
        valid
    ) == 0:

        continue


    print(

        f"{column:<42} "
        f"min {valid.min():+7.3f} | "
        f"mean {valid.mean():+7.3f} | "
        f"max {valid.max():+7.3f}"
    )


# ============================================================
# EXAMPLE HIGH-INSTABILITY TEAMS/GAMES
#
# This is only an audit so we can confirm the values look
# sensible.
# ============================================================

examples = features_v4[
    [
        "season",
        "week",
        "home_team",
        "away_team",
        "qb_home_distinct_primary_passers",
        "qb_away_distinct_primary_passers",
        "qb_home_primary_streak",
        "qb_away_primary_streak",
        "qb_home_qb_change_rate",
        "qb_away_qb_change_rate"
    ]
].copy()


examples[
    "maximum_change_rate"
] = examples[
    [
        "qb_home_qb_change_rate",
        "qb_away_qb_change_rate"
    ]
].max(
    axis=1
)


examples = (

    examples

    .dropna(
        subset=[
            "maximum_change_rate"
        ]
    )

    .sort_values(
        [
            "maximum_change_rate",
            "season",
            "week"
        ],

        ascending=[
            False,
            False,
            False
        ]
    )

    .head(
        15
    )
)


print()

print(
    "=" * 110
)

print(
    "SAMPLE HIGH-QB-INSTABILITY MATCHUPS"
)

print(
    "=" * 110
)


print(

    examples.to_string(

        index=False,

        formatters={

            "qb_home_qb_change_rate":
                lambda x:
                    f"{x:.2%}",

            "qb_away_qb_change_rate":
                lambda x:
                    f"{x:.2%}",

            "maximum_change_rate":
                lambda x:
                    f"{x:.2%}"
        }
    )
)


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    "data/processed",
    exist_ok=True
)


features_v4.to_csv(
    OUTPUT_FILE,
    index=False
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


print(
    f"Rows:    "
    f"{len(features_v4):,}"
)


print(
    f"Columns: "
    f"{len(features_v4.columns):,}"
)


print(
    f"New QB columns: "
    f"{len(qb_frame.columns):,}"
)


print(
    f"Saved:   "
    f"{OUTPUT_FILE}"
)


print()

print(
    "Current-game passer information was NEVER "
    "used to construct that game's features."
)