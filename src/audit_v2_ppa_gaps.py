import sqlite3
import pandas as pd


# ==================================================
# SETTINGS
# ==================================================

database_file = "data/cfb.db"

invalid_game_ids = {
    401643737
}


# ==================================================
# PLAY DEFINITIONS
# ==================================================

rush_types = {
    "Rush",
    "Rushing Touchdown"
}


dropback_types = {

    "Pass",
    "Pass Reception",
    "Pass Incompletion",
    "Passing Touchdown",

    "Sack",

    "Interception",
    "Pass Interception Return",
    "Interception Return Touchdown"
}


other_scrimmage_types = {

    "Fumble Recovery (Own)",
    "Fumble Recovery (Opponent)",

    "Fumble Return Touchdown",

    "Safety"
}


core_types = (
    rush_types
    |
    dropback_types
    |
    other_scrimmage_types
)


# ==================================================
# CONNECT
# ==================================================

connection = sqlite3.connect(
    database_file
)


print("V2 PPA GAP AUDIT")
print("=" * 110)


# ==================================================
# LOAD 2024 FBS-V-FBS GAMES
# ==================================================

games_query = """
SELECT

    game_id,
    season,
    week,
    start_date,

    away_team,
    away_points,

    home_team,
    home_points

FROM games

WHERE
    season = 2024

    AND home_points IS NOT NULL
    AND away_points IS NOT NULL

    AND home_classification = 'fbs'
    AND away_classification = 'fbs'

ORDER BY
    week,
    start_date,
    game_id
"""


games = pd.read_sql_query(
    games_query,
    connection
)


print(
    f"2024 FBS-v-FBS games: "
    f"{len(games):,}"
)


# ==================================================
# LOAD ALL PPA PLAYS FOR 2024
# ==================================================

plays_query = """
SELECT

    p.game_id,
    p.offense,
    p.defense,
    p.play_type,
    p.ppa

FROM plays AS p

JOIN games AS g
    ON p.game_id = g.game_id

WHERE
    g.season = 2024

    AND g.home_points IS NOT NULL
    AND g.away_points IS NOT NULL

    AND g.home_classification = 'fbs'
    AND g.away_classification = 'fbs'

    AND p.ppa IS NOT NULL
"""


plays = pd.read_sql_query(
    plays_query,
    connection
)


connection.close()


# ==================================================
# CORE SCRIMMAGE PLAYS
# ==================================================

core_plays = plays[
    plays[
        "play_type"
    ].isin(
        core_types
    )
].copy()


# ==================================================
# AUDIT EACH GAME
# ==================================================

results = []


for _, game in games.iterrows():

    game_id = game[
        "game_id"
    ]

    home_team = game[
        "home_team"
    ]

    away_team = game[
        "away_team"
    ]


    all_game_plays = plays[
        plays[
            "game_id"
        ] == game_id
    ]


    game_core = core_plays[
        core_plays[
            "game_id"
        ] == game_id
    ]


    home_offense = (
        game_core[
            game_core[
                "offense"
            ] == home_team
        ]
    )


    away_offense = (
        game_core[
            game_core[
                "offense"
            ] == away_team
        ]
    )


    home_defense = (
        game_core[
            game_core[
                "defense"
            ] == home_team
        ]
    )


    away_defense = (
        game_core[
            game_core[
                "defense"
            ] == away_team
        ]
    )


    usable = (

        len(home_offense) > 0
        and
        len(away_offense) > 0

        and
        len(home_defense) > 0
        and
        len(away_defense) > 0

        and
        game_id not in invalid_game_ids
    )


    offense_labels = sorted(
        [
            str(value)

            for value in
            all_game_plays[
                "offense"
            ]
            .dropna()
            .unique()
        ]
    )


    defense_labels = sorted(
        [
            str(value)

            for value in
            all_game_plays[
                "defense"
            ]
            .dropna()
            .unique()
        ]
    )


    results.append(
        {
            "game_id":
                game_id,

            "week":
                game["week"],

            "away_team":
                away_team,

            "home_team":
                home_team,

            "total_ppa_plays":
                len(
                    all_game_plays
                ),

            "core_ppa_plays":
                len(
                    game_core
                ),

            "home_off_plays":
                len(
                    home_offense
                ),

            "away_off_plays":
                len(
                    away_offense
                ),

            "home_def_plays":
                len(
                    home_defense
                ),

            "away_def_plays":
                len(
                    away_defense
                ),

            "offense_labels":
                " | ".join(
                    offense_labels
                ),

            "defense_labels":
                " | ".join(
                    defense_labels
                ),

            "known_partial_game":
                (
                    game_id
                    in invalid_game_ids
                ),

            "usable":
                usable
        }
    )


results = pd.DataFrame(
    results
)


# ==================================================
# SHOW GAPS
# ==================================================

gaps = results[
    ~results[
        "usable"
    ]
].copy()


print("\n")
print("2024 GAMES NOT USABLE FOR V2 PPA")
print("=" * 110)

print(
    f"Count: "
    f"{len(gaps):,}"
)


columns = [

    "game_id",
    "week",

    "away_team",
    "home_team",

    "total_ppa_plays",
    "core_ppa_plays",

    "home_off_plays",
    "away_off_plays",

    "home_def_plays",
    "away_def_plays",

    "known_partial_game"
]


print(
    gaps[
        columns
    ].to_string(
        index=False
    )
)


# ==================================================
# SHOW RAW TEAM LABELS
# ==================================================

print("\n")
print("RAW PLAY-BY-PLAY TEAM LABELS")
print("=" * 110)


for _, row in gaps.iterrows():

    print("\n")
    print("-" * 110)

    print(
        f"Game ID: "
        f"{row['game_id']}"
    )

    print(
        f"Week: "
        f"{row['week']}"
    )

    print(
        f"Game table: "
        f"{row['away_team']} "
        f"@ "
        f"{row['home_team']}"
    )

    print(
        f"Play offense labels: "
        f"{row['offense_labels']}"
    )

    print(
        f"Play defense labels: "
        f"{row['defense_labels']}"
    )

    print(
        f"Total PPA plays: "
        f"{row['total_ppa_plays']}"
    )

    print(
        f"Core PPA plays: "
        f"{row['core_ppa_plays']}"
    )

    print(
        f"Known partial game: "
        f"{row['known_partial_game']}"
    )


# ==================================================
# SPECIFIC MISMATCH CHECK
# ==================================================

print("\n")
print("TEAM NAME MISMATCH CHECK")
print("=" * 110)


for _, row in gaps.iterrows():

    expected = {
        row[
            "home_team"
        ],
        row[
            "away_team"
        ]
    }


    observed_offense = set(
        row[
            "offense_labels"
        ].split(
            " | "
        )
    )


    observed_defense = set(
        row[
            "defense_labels"
        ].split(
            " | "
        )
    )


    missing_from_offense = (
        expected
        -
        observed_offense
    )


    missing_from_defense = (
        expected
        -
        observed_defense
    )


    print("\n")

    print(
        f"{row['away_team']} "
        f"@ "
        f"{row['home_team']}"
    )

    print(
        "Expected team names:"
    )

    print(
        expected
    )

    print(
        "Missing from offense labels:"
    )

    print(
        missing_from_offense
    )

    print(
        "Missing from defense labels:"
    )

    print(
        missing_from_defense
    )


# ==================================================
# SUMMARY
# ==================================================

print("\n")
print("SUMMARY")
print("=" * 110)

print(
    f"2024 games:              "
    f"{len(results):,}"
)

print(
    f"Usable PPA games:        "
    f"{results['usable'].sum():,}"
)

print(
    f"Unusable PPA games:      "
    f"{len(gaps):,}"
)

print(
    f"Known partial games:     "
    f"{gaps['known_partial_game'].sum():,}"
)

print(
    f"Other unexplained gaps:  "
    f"{len(gaps) - gaps['known_partial_game'].sum():,}"
)