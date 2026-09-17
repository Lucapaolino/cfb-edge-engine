import os
import sqlite3
from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIRECTORY = PROJECT_ROOT / "data"

DATABASE_FILE = DATA_DIRECTORY / "cfb.db"


# ============================================================
# START
# ============================================================

print(
    "LIVE MODEL READINESS AUDIT — V1"
)

print(
    "=" * 110
)


# ============================================================
# MODEL ARTIFACT SEARCH
#
# Look for saved PyTorch / sklearn models.
# ============================================================

print()

print(
    "SAVED MODEL ARTIFACTS"
)

print(
    "-" * 110
)


model_extensions = [

    "*.pt",
    "*.pth",
    "*.pkl",
    "*.pickle",
    "*.joblib"
]


model_files = []


for pattern in model_extensions:

    model_files.extend(
        PROJECT_ROOT.rglob(
            pattern
        )
    )


model_files = sorted(
    set(
        model_files
    )
)


if len(
    model_files
) == 0:

    print(
        "No saved model artifacts found."
    )


else:

    for file in model_files:

        relative = file.relative_to(
            PROJECT_ROOT
        )


        size_mb = (
            file.stat().st_size
            /
            1024
            /
            1024
        )


        print(
            f"{relative} "
            f"({size_mb:.2f} MB)"
        )


# ============================================================
# FEATURE FILES
# ============================================================

print()

print(
    "FEATURE DATASETS"
)

print(
    "-" * 110
)


feature_files = sorted(

    DATA_DIRECTORY.rglob(
        "features*.csv"
    )
)


if len(
    feature_files
) == 0:

    print(
        "No features*.csv files found."
    )


else:

    for file in feature_files:

        try:

            dataframe = pd.read_csv(
                file
            )


            relative = file.relative_to(
                PROJECT_ROOT
            )


            print()

            print(
                relative
            )


            print(
                f"  rows:    "
                f"{len(dataframe):,}"
            )


            print(
                f"  columns: "
                f"{len(dataframe.columns):,}"
            )


            if "season" in dataframe.columns:

                seasons = (

                    pd.to_numeric(
                        dataframe[
                            "season"
                        ],
                        errors="coerce"
                    )

                    .dropna()

                    .astype(
                        int
                    )

                    .unique()
                )


                seasons = sorted(
                    seasons
                )


                print(
                    f"  seasons: "
                    f"{seasons}"
                )


                season_2026 = dataframe[

                    pd.to_numeric(
                        dataframe[
                            "season"
                        ],
                        errors="coerce"
                    )
                    ==
                    2026
                ]


                print(
                    f"  2026 rows: "
                    f"{len(season_2026):,}"
                )


        except Exception as error:

            print(
                f"{file}: ERROR: {error}"
            )


# ============================================================
# DATABASE
# ============================================================

print()

print(
    "=" * 110
)

print(
    "DATABASE CHECK"
)

print(
    "=" * 110
)


if not DATABASE_FILE.exists():

    raise RuntimeError(
        f"Database not found: "
        f"{DATABASE_FILE}"
    )


connection = sqlite3.connect(
    DATABASE_FILE
)


tables = pd.read_sql_query(
    """
    SELECT name

    FROM sqlite_master

    WHERE type = 'table'

    ORDER BY name
    """,
    connection
)


print()

print(
    "TABLES"
)

print(
    "-" * 110
)


for table in tables[
    "name"
]:

    print(
        table
    )


# ============================================================
# GAMES TABLE
# ============================================================

table_names = set(
    tables[
        "name"
    ]
)


if "games" in table_names:

    print()

    print(
        "2026 GAME COVERAGE"
    )

    print(
        "-" * 110
    )


    game_columns = pd.read_sql_query(
        """
        PRAGMA table_info(games)
        """,
        connection
    )


    game_column_names = set(
        game_columns[
            "name"
        ]
    )


    print(
        "Game columns:"
    )


    print(
        ", ".join(
            sorted(
                game_column_names
            )
        )
    )


    if "season" in game_column_names:

        count_2026 = connection.execute(
            """
            SELECT COUNT(*)

            FROM games

            WHERE season = 2026
            """
        ).fetchone()[0]


        print()

        print(
            f"2026 games stored: "
            f"{count_2026:,}"
        )


        score_columns = {

            "home_points",
            "away_points"
        }


        if score_columns.issubset(
            game_column_names
        ):

            completed_2026 = connection.execute(
                """
                SELECT COUNT(*)

                FROM games

                WHERE
                    season = 2026
                    AND home_points IS NOT NULL
                    AND away_points IS NOT NULL
                """
            ).fetchone()[0]


            upcoming_2026 = connection.execute(
                """
                SELECT COUNT(*)

                FROM games

                WHERE
                    season = 2026
                    AND (
                        home_points IS NULL
                        OR away_points IS NULL
                    )
                """
            ).fetchone()[0]


            print(
                f"2026 completed:    "
                f"{completed_2026:,}"
            )


            print(
                f"2026 upcoming:     "
                f"{upcoming_2026:,}"
            )


# ============================================================
# TEAM STATS
# ============================================================

if "team_game_stats" in table_names:

    count = connection.execute(
        """
        SELECT COUNT(*)

        FROM team_game_stats
        """
    ).fetchone()[0]


    print()

    print(
        f"Team stat rows total: "
        f"{count:,}"
    )


# ============================================================
# QB / PLAYER PASSING
# ============================================================

if "player_game_stat_values" in table_names:

    print()

    print(
        "PASSING DATA"
    )

    print(
        "-" * 110
    )


    columns = pd.read_sql_query(
        """
        PRAGMA table_info(player_game_stat_values)
        """,
        connection
    )


    column_names = set(
        columns[
            "name"
        ]
    )


    if {
        "category",
        "stat_type"
    }.issubset(
        column_names
    ):

        passing_rows = connection.execute(
            """
            SELECT COUNT(*)

            FROM player_game_stat_values

            WHERE LOWER(category) = 'passing'
            """
        ).fetchone()[0]


        print(
            f"Passing stat rows: "
            f"{passing_rows:,}"
        )


# ============================================================
# LIVE MARKET
# ============================================================

if "execution_market_snapshots" in table_names:

    print()

    print(
        "LIVE MARKET DATA"
    )

    print(
        "-" * 110
    )


    batches = connection.execute(
        """
        SELECT COUNT(
            DISTINCT batch_id
        )

        FROM execution_market_snapshots
        """
    ).fetchone()[0]


    rows = connection.execute(
        """
        SELECT COUNT(*)

        FROM execution_market_snapshots
        """
    ).fetchone()[0]


    print(
        f"Market batches: "
        f"{batches:,}"
    )


    print(
        f"Market rows:    "
        f"{rows:,}"
    )


connection.close()


# ============================================================
# TRAINING SCRIPTS
# ============================================================

print()

print(
    "=" * 110
)

print(
    "MODEL / FEATURE SCRIPTS"
)

print(
    "=" * 110
)


important_scripts = [

    "train_neural_network_v1.py",

    "build_features_v1.py",

    "build_features_v2.py",

    "build_opponent_adjusted_features_v1.py",

    "build_qb_continuity_features_v1.py",

    "test_market_aware_win_model_v1.py",

    "test_qb_market_residual_v1.py"
]


for script in important_scripts:

    path = (
        PROJECT_ROOT
        /
        "src"
        /
        script
    )


    status = (
        "FOUND"
        if path.exists()
        else "MISSING"
    )


    print(
        f"{status:8}  {script}"
    )


# ============================================================
# RESULT
# ============================================================

print()

print(
    "=" * 110
)

print(
    "READINESS INTERPRETATION"
)

print(
    "=" * 110
)


if len(
    model_files
) == 0:

    print(
        "1. We do not appear to have a permanently saved "
        "trained model yet."
    )


    print(
        "   Next step will be to create and freeze one "
        "for live inference."
    )


else:

    print(
        "1. Saved model artifact(s) exist."
    )


    print(
        "   We can inspect which one should power "
        "live inference."
    )


print()

print(
    "2. We need a 2026 pregame feature row for each "
    "upcoming game before the neural network can score it."
)


print()

print(
    "3. Once those two pieces are ready, the GUI pipeline "
    "gets a real NEURAL INFERENCE stage."
)


print()

print(
    "=" * 110
)

print(
    "AUDIT COMPLETE"
)

print(
    "=" * 110
)