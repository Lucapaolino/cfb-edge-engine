import os

import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = (
    "data/processed/"
    "features_v2.csv"
)

OUTPUT_FILE = (
    "data/processed/"
    "features_v3.csv"
)


# ------------------------------------------------------------
# Ridge shrinkage.
#
# Higher = team ratings are pulled more strongly toward zero.
#
# We are fixing this at 20 rather than tuning it against
# betting results.
# ------------------------------------------------------------

TEAM_RIDGE_ALPHA = 20.0


# ------------------------------------------------------------
# Much lighter penalty on the estimated home-field effect.
# ------------------------------------------------------------

HOME_FIELD_RIDGE_ALPHA = 1.0


# ============================================================
# START
# ============================================================

print(
    "BUILD OPPONENT-ADJUSTED FEATURES — V1"
)

print(
    "=" * 110
)


data = pd.read_csv(
    INPUT_FILE
)


print(
    f"Rows loaded:    {len(data):,}"
)

print(
    f"Columns loaded: {len(data.columns):,}"
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "game_id",
    "season",
    "week",
    "home_team",
    "away_team",
    "neutral_site",
    "target_home_margin"
]


missing = [
    column
    for column in required_columns
    if column not in data.columns
]


if missing:

    raise RuntimeError(
        "Missing required columns: "
        +
        ", ".join(missing)
    )


# ============================================================
# CLEAN BASIC FIELDS
# ============================================================

data["season"] = pd.to_numeric(
    data["season"],
    errors="coerce"
)


data["week"] = pd.to_numeric(
    data["week"],
    errors="coerce"
)


data["target_home_margin"] = pd.to_numeric(
    data["target_home_margin"],
    errors="coerce"
)


data["neutral_site"] = pd.to_numeric(
    data["neutral_site"],
    errors="coerce"
).fillna(0)


# ============================================================
# OUTPUT ARRAYS
#
# NaN is deliberate.
#
# Week 1 has no current-season history, so opponent-adjusted
# strength should not magically exist.
# ============================================================

home_rating_output = np.full(
    len(data),
    np.nan
)

away_rating_output = np.full(
    len(data),
    np.nan
)

rating_diff_output = np.full(
    len(data),
    np.nan
)

expected_margin_output = np.full(
    len(data),
    np.nan
)

home_field_output = np.full(
    len(data),
    np.nan
)

home_games_output = np.zeros(
    len(data),
    dtype=float
)

away_games_output = np.zeros(
    len(data),
    dtype=float
)

available_output = np.zeros(
    len(data),
    dtype=int
)


# ============================================================
# FIT OPPONENT-ADJUSTED RATINGS
# ============================================================

def fit_strength_ratings(
    history,
    all_season_teams
):

    # --------------------------------------------------------
    # Each team gets one coefficient.
    #
    # A historical game is represented as:
    #
    # +1 for home team
    # -1 for away team
    #
    # plus a home-field indicator.
    #
    # Target = actual home point margin.
    # --------------------------------------------------------

    teams = sorted(
        all_season_teams
    )


    team_to_index = {
        team: index
        for index, team
        in enumerate(teams)
    }


    number_of_teams = len(
        teams
    )


    number_of_games = len(
        history
    )


    # Last column is home-field advantage.
    x = np.zeros(
        (
            number_of_games,
            number_of_teams + 1
        ),
        dtype=float
    )


    y = history[
        "target_home_margin"
    ].to_numpy(
        dtype=float
    )


    for row_number, (_, game) in enumerate(
        history.iterrows()
    ):

        home_team = game[
            "home_team"
        ]

        away_team = game[
            "away_team"
        ]


        x[
            row_number,
            team_to_index[
                home_team
            ]
        ] = 1.0


        x[
            row_number,
            team_to_index[
                away_team
            ]
        ] = -1.0


        # ----------------------------------------------------
        # Home-field coefficient applies only to
        # non-neutral games.
        # ----------------------------------------------------

        if int(
            game[
                "neutral_site"
            ]
        ) == 0:

            x[
                row_number,
                -1
            ] = 1.0


    # ========================================================
    # RIDGE SOLUTION
    #
    # beta =
    # (X'X + penalty)^-1 X'y
    #
    # Ridge is useful here because early in a season there
    # are relatively few games and team schedules can be
    # poorly connected.
    # ========================================================

    penalty = np.eye(
        number_of_teams + 1
    )


    penalty[
        :number_of_teams,
        :number_of_teams
    ] *= TEAM_RIDGE_ALPHA


    penalty[
        -1,
        -1
    ] = HOME_FIELD_RIDGE_ALPHA


    lhs = (
        x.T @ x
        +
        penalty
    )


    rhs = (
        x.T @ y
    )


    coefficients = np.linalg.solve(
        lhs,
        rhs
    )


    ratings = {
        team:
            float(
                coefficients[
                    team_to_index[
                        team
                    ]
                ]
            )
        for team in teams
    }


    home_field = float(
        coefficients[
            -1
        ]
    )


    return (
        ratings,
        home_field
    )


# ============================================================
# PROCESS ONE SEASON AT A TIME
# ============================================================

for season in sorted(
    data[
        "season"
    ].dropna().unique()
):

    season = int(
        season
    )


    season_mask = (
        data[
            "season"
        ]
        ==
        season
    )


    season_data = data[
        season_mask
    ].copy()


    season_teams = set(
        season_data[
            "home_team"
        ].dropna()
    )


    season_teams.update(
        season_data[
            "away_team"
        ].dropna()
    )


    weeks = sorted(
        season_data[
            "week"
        ].dropna().unique()
    )


    print()

    print(
        f"Season {season}"
    )

    print(
        "-" * 70
    )


    season_available_games = 0


    for week in weeks:

        # ====================================================
        # HISTORY WALL
        #
        # Only games from PREVIOUS weeks are allowed.
        #
        # Current-week outcomes can never affect features.
        # ====================================================

        history = season_data[
            (
                season_data[
                    "week"
                ]
                <
                week
            )
            &
            (
                season_data[
                    "target_home_margin"
                ].notna()
            )
        ].copy()


        current_week = season_data[
            season_data[
                "week"
            ]
            ==
            week
        ].copy()


        if len(
            history
        ) == 0:

            # Week 1 normally lands here.
            continue


        (
            ratings,
            home_field
        ) = fit_strength_ratings(
            history,
            season_teams
        )


        # ----------------------------------------------------
        # Count how many prior games each team has played.
        # ----------------------------------------------------

        prior_game_counts = {}


        for team in season_teams:

            count = (

                (
                    history[
                        "home_team"
                    ]
                    ==
                    team
                )

                |

                (
                    history[
                        "away_team"
                    ]
                    ==
                    team
                )

            ).sum()


            prior_game_counts[
                team
            ] = int(
                count
            )


        # ====================================================
        # ASSIGN CURRENT-WEEK FEATURES
        # ====================================================

        for index, game in current_week.iterrows():

            home_team = game[
                "home_team"
            ]

            away_team = game[
                "away_team"
            ]


            home_games = prior_game_counts.get(
                home_team,
                0
            )


            away_games = prior_game_counts.get(
                away_team,
                0
            )


            home_games_output[
                index
            ] = home_games


            away_games_output[
                index
            ] = away_games


            # ------------------------------------------------
            # If one team has played no previous game this
            # season, don't pretend its rating is meaningful.
            # ------------------------------------------------

            if (
                home_games == 0
                or
                away_games == 0
            ):

                continue


            home_rating = ratings.get(
                home_team,
                0.0
            )


            away_rating = ratings.get(
                away_team,
                0.0
            )


            rating_diff = (
                home_rating
                -
                away_rating
            )


            if int(
                game[
                    "neutral_site"
                ]
            ) == 0:

                expected_margin = (
                    rating_diff
                    +
                    home_field
                )

            else:

                expected_margin = (
                    rating_diff
                )


            home_rating_output[
                index
            ] = home_rating


            away_rating_output[
                index
            ] = away_rating


            rating_diff_output[
                index
            ] = rating_diff


            expected_margin_output[
                index
            ] = expected_margin


            home_field_output[
                index
            ] = home_field


            available_output[
                index
            ] = 1


            season_available_games += 1


    print(
        f"Games with opponent-adjusted ratings: "
        f"{season_available_games:,} / "
        f"{len(season_data):,}"
    )


# ============================================================
# ADD FEATURES
# ============================================================

data[
    "oa_home_strength"
] = home_rating_output


data[
    "oa_away_strength"
] = away_rating_output


data[
    "oa_strength_diff"
] = rating_diff_output


data[
    "oa_expected_margin"
] = expected_margin_output


data[
    "oa_home_field_estimate"
] = home_field_output


data[
    "oa_home_prior_games"
] = home_games_output


data[
    "oa_away_prior_games"
] = away_games_output


data[
    "oa_available"
] = available_output


# ============================================================
# AUDIT
# ============================================================

print()

print(
    "=" * 110
)

print(
    "OPPONENT-ADJUSTED FEATURE AUDIT"
)

print(
    "=" * 110
)


audit = (

    data

    .groupby(
        "season"
    )

    .agg(

        games=(
            "game_id",
            "size"
        ),

        available=(
            "oa_available",
            "sum"
        ),

        average_expected_margin=(
            "oa_expected_margin",
            "mean"
        ),

        average_home_field=(
            "oa_home_field_estimate",
            "mean"
        )
    )

    .reset_index()
)


audit[
    "coverage"
] = (

    audit[
        "available"
    ]

    /

    audit[
        "games"
    ]
)


print(

    audit.to_string(

        index=False,

        formatters={

            "coverage":
                lambda x:
                    f"{x:.2%}",

            "average_expected_margin":
                lambda x:
                    f"{x:+.3f}",

            "average_home_field":
                lambda x:
                    f"{x:+.3f}"
        }
    )
)


# ============================================================
# LEAKAGE AUDIT
#
# Week 1 MUST have zero opponent-adjusted availability.
# ============================================================

week_one = data[
    data[
        "week"
    ]
    ==
    1
]


week_one_available = int(
    week_one[
        "oa_available"
    ].sum()
)


print()

print(
    f"Week 1 rows with ratings available: "
    f"{week_one_available}"
)


if week_one_available != 0:

    raise RuntimeError(
        "Leakage warning: Week 1 has "
        "opponent-adjusted ratings."
    )


# ============================================================
# SIMPLE VALUE RANGES
# ============================================================

available = data[
    data[
        "oa_available"
    ]
    ==
    1
]


print()

print(
    "AVAILABLE FEATURE RANGES"
)

print(
    "-" * 70
)


for column in [
    "oa_home_strength",
    "oa_away_strength",
    "oa_strength_diff",
    "oa_expected_margin",
    "oa_home_field_estimate"
]:

    print(

        f"{column:<28} "
        f"min {available[column].min():+8.3f} | "
        f"mean {available[column].mean():+8.3f} | "
        f"max {available[column].max():+8.3f}"
    )


# ============================================================
# SAVE V3
# ============================================================

os.makedirs(
    "data/processed",
    exist_ok=True
)


data.to_csv(
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
    f"Rows:    {len(data):,}"
)


print(
    f"Columns: {len(data.columns):,}"
)


print(
    f"Saved:   {OUTPUT_FILE}"
)


print()

print(
    "No current-week result was used to create "
    "its opponent-adjusted features."
)