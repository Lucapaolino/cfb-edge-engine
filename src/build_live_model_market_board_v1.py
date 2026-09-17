import re
import sqlite3
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PURPOSE
#
# Combine:
#   1. Frozen neural-network predictions
#   2. Hard Rock Bet execution lines from OddsPapi
#   3. Multi-book reference consensus from The Odds API
#
# No APIs are called by this script.
#
# IMPORTANT SAFETY / QUALITY RULES
#
# - Hard Rock line is usable only when spread_quality_pass = 1.
# - Rejected alternate/truncated ladders never create MODEL EDGE.
# - Model/market disagreement >= 8 points forces Strength LOW.
# - Model/market disagreement >= 6 points caps Strength at MEDIUM.
# - Model edge < 1 point caps Strength at LOW.
# - Started/past games are not actionable.
# - Matching fails closed when ambiguous.
# ============================================================


# ============================================================
# SETTINGS
# ============================================================

SPORT = "americanfootball_ncaaf"

REFERENCE_MIN_BOOKS = 3

HARDROCK_MAX_SNAPSHOT_AGE_HOURS = 2.0
REFERENCE_MAX_SNAPSHOT_AGE_HOURS = 2.0

MAX_EVENT_TIME_DIFFERENCE_HOURS = 8.0
EXACT_PAIR_MAX_TIME_DIFFERENCE_HOURS = 18.0
MIN_EVENT_TEAM_SCORE = 0.74
AMBIGUOUS_MATCH_MARGIN = 0.03

LARGE_MODEL_MARKET_GAP = 6.0
OUTLIER_MODEL_MARKET_GAP = 8.0
MIN_MEANINGFUL_MODEL_EDGE = 1.0


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


DATABASE_FILE = (
    PROJECT_ROOT
    / "data"
    / "cfb.db"
)


PREDICTION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "live_nn_predictions_2026_v1.csv"
)


HARDROCK_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hardrock_execution_latest.csv"
)


OUTPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "live_model_market_board_2026_v1.csv"
)


OUTPUT_JSON = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "live_model_market_board_2026_v1.json"
)


# ============================================================
# GENERAL HELPERS
# ============================================================

def header(text):

    print()

    print(
        "=" * 110
    )

    print(
        text
    )

    print(
        "=" * 110
    )


def safe_float(value):

    try:

        if (
            value is None
            or
            pd.isna(
                value
            )
        ):

            return np.nan


        return float(
            value
        )


    except Exception:

        return np.nan


def safe_int(
    value,
    default=0
):

    try:

        if (
            value is None
            or
            pd.isna(
                value
            )
        ):

            return default


        return int(
            value
        )


    except Exception:

        return default


def american_price_display(
    price
):

    if (
        price is None
        or
        pd.isna(
            price
        )
    ):

        return ""


    price = int(
        round(
            float(
                price
            )
        )
    )


    if price > 0:

        return (
            f"+{price}"
        )


    return str(
        price
    )


def spread_display(
    team,
    spread,
    price=None
):

    if (
        spread is None
        or
        pd.isna(
            spread
        )
    ):

        return "Unavailable"


    spread = float(
        spread
    )


    if abs(
        spread
    ) < 0.05:

        spread_text = "PK"


    elif spread > 0:

        spread_text = (
            f"+{spread:.1f}"
        )


    else:

        spread_text = (
            f"{spread:.1f}"
        )


    result = (

        f"{team} "
        f"{spread_text}"
    )


    if (
        price is not None
        and
        pd.notna(
            price
        )
    ):

        result += (

            f" ("
            f"{american_price_display(price)}"
            f")"
        )


    return result


def edge_display(
    team,
    edge
):

    if (
        edge is None
        or
        pd.isna(
            edge
        )
    ):

        return "N/A"


    edge = float(
        edge
    )


    if abs(
        edge
    ) < 0.05:

        return "No edge"


    if edge > 0:

        return (

            f"{team} "
            f"+{edge:.1f} pts"
        )


    return (

        f"No edge "
        f"({team} "
        f"{edge:.1f} pts)"
    )


# ============================================================
# TEAM NORMALIZATION / MATCHING
# ============================================================

TEAM_ALIASES = {

    "uconn":
        "connecticut",

    "connecticut huskies":
        "connecticut",

    "pitt":
        "pittsburgh",

    "pittsburgh panthers":
        "pittsburgh",

    "fiu":
        "florida international",

    "florida international panthers":
        "florida international",

    "umass":
        "massachusetts",

    "massachusetts minutemen":
        "massachusetts",

    "ul monroe":
        "louisiana monroe",

    "ulm":
        "louisiana monroe",

    "louisiana monroe warhawks":
        "louisiana monroe",

    "western kentucky hilltoppers":
        "western kentucky",

    "wku":
        "western kentucky",

    "hawai i":
        "hawaii",

    "hawaii rainbow warriors":
        "hawaii",

    "sam houston state":
        "sam houston",

    "sam houston bearkats":
        "sam houston",

    "louisiana ragin cajuns":
        "louisiana",

    "ragin cajuns":
        "louisiana",

    "louisiana lafayette":
        "louisiana",

    "usc trojans":
        "usc",

    "southern california":
        "usc",

    "ucf knights":
        "ucf",

    "byu cougars":
        "byu",

    "brigham young":
        "byu"
}


# These words distinguish schools and should not be treated
# like disposable mascot suffixes.

STRUCTURAL_TEAM_WORDS = {

    "state",
    "tech",
    "southern",
    "northern",
    "eastern",
    "western",
    "central",
    "international",
    "atlantic",
    "christian",
    "a",
    "m",
    "am",
    "poly",
    "valley"
}


def normalize_team_name(
    name
):

    if name is None:

        return ""


    text = (
        str(
            name
        )
        .strip()
        .lower()
    )


    text = (
        text
        .replace(
            "'",
            ""
        )
        .replace(
            "’",
            ""
        )
    )


    text = unicodedata.normalize(
        "NFKD",
        text
    )


    text = "".join(

        character

        for character in text

        if not unicodedata.combining(
            character
        )
    )


    text = re.sub(

        r"[^a-z0-9]+",

        " ",

        text
    )


    text = re.sub(

        r"\s+",

        " ",

        text
    ).strip()


    return TEAM_ALIASES.get(
        text,
        text
    )


def _single_token_prefix_is_safe(
    shorter,
    longer
):

    shorter_tokens = shorter.split()

    longer_tokens = longer.split()


    if len(
        shorter_tokens
    ) != 1:

        return True


    if (
        not longer_tokens
        or
        longer_tokens[
            0
        ]
        !=
        shorter_tokens[
            0
        ]
    ):

        return False


    extra_tokens = set(

        longer_tokens[
            1:
        ]
    )


    if (
        extra_tokens

        &

        STRUCTURAL_TEAM_WORDS
    ):

        return False


    return True


def team_similarity(
    left,
    right
):

    left = normalize_team_name(
        left
    )


    right = normalize_team_name(
        right
    )


    if not left or not right:

        return 0.0


    if left == right:

        return 1.0


    left_tokens = left.split()

    right_tokens = right.split()


    if len(
        left_tokens
    ) <= len(
        right_tokens
    ):

        shorter = left

        longer = right

        shorter_tokens = left_tokens

        longer_tokens = right_tokens


    else:

        shorter = right

        longer = left

        shorter_tokens = right_tokens

        longer_tokens = left_tokens


    # --------------------------------------------------------
    # Common provider pattern:
    #
    # Alabama
    # Alabama Crimson Tide
    #
    # But avoid:
    #
    # Texas
    # Texas State
    #
    # Ohio
    # Ohio State
    # --------------------------------------------------------

    if (

        longer_tokens[
            :len(
                shorter_tokens
            )
        ]

        ==

        shorter_tokens
    ):

        if len(
            shorter_tokens
        ) >= 2:

            return 0.98


        if _single_token_prefix_is_safe(

            shorter,

            longer
        ):

            return 0.96


    # --------------------------------------------------------
    # Token overlap
    # --------------------------------------------------------

    left_set = set(
        left_tokens
    )


    right_set = set(
        right_tokens
    )


    intersection = len(

        left_set

        &

        right_set
    )


    union = len(

        left_set

        |

        right_set
    )


    token_score = (

        intersection
        /
        union

        if union > 0

        else 0.0
    )


    sequence_score = SequenceMatcher(

        None,

        left,

        right

    ).ratio()


    return max(

        sequence_score,

        0.85
        *
        token_score
    )


def best_similarity(
    team,
    *names
):

    scores = [

        team_similarity(
            team,
            name
        )

        for name in names

        if name is not None
    ]


    if not scores:

        return 0.0


    return max(
        scores
    )


def event_pair_scores(

    prediction_home,

    prediction_away,

    vendor_team_1,

    vendor_team_2
):

    normal_home = team_similarity(

        prediction_home,

        vendor_team_1
    )


    normal_away = team_similarity(

        prediction_away,

        vendor_team_2
    )


    swapped_home = team_similarity(

        prediction_home,

        vendor_team_2
    )


    swapped_away = team_similarity(

        prediction_away,

        vendor_team_1
    )


    normal = (

        normal_home
        +
        normal_away

    ) / 2.0


    swapped = (

        swapped_home
        +
        swapped_away

    ) / 2.0


    return (
        normal,
        swapped
    )


def find_best_event_match(

    prediction_row,

    events,

    event_id_column,

    event_time_column,

    team1_column,

    team2_column
):

    game_time = prediction_row[
        "start_date"
    ]


    if (
        pd.isna(
            game_time
        )
        or
        len(
            events
        )
        ==
        0
    ):

        return None


    candidates = []


    for _, event in events.iterrows():

        event_time = event[
            event_time_column
        ]


        if pd.isna(
            event_time
        ):

            continue


        (
            normal_score,
            swapped_score

        ) = event_pair_scores(

            prediction_row[
                "home_team"
            ],

            prediction_row[
                "away_team"
            ],

            event[
                team1_column
            ],

            event[
                team2_column
            ]
        )


        team_score = max(

            normal_score,

            swapped_score
        )


        if team_score < MIN_EVENT_TEAM_SCORE:

            continue


        time_difference = abs(

            (
                event_time
                -
                game_time
            ).total_seconds()

        ) / 3600.0


        exactish_pair = (

            team_score

            >=

            0.96
        )


        allowed_hours = (

            EXACT_PAIR_MAX_TIME_DIFFERENCE_HOURS

            if exactish_pair

            else MAX_EVENT_TIME_DIFFERENCE_HOURS
        )


        if time_difference > allowed_hours:

            continue


        time_score = max(

            0.0,

            1.0

            -

            (
                time_difference
                /
                allowed_hours
            )
        )


        total_score = (

            0.94
            *
            team_score

            +

            0.06
            *
            time_score
        )


        candidates.append(

            {
                "event_id":
                    event[
                        event_id_column
                    ],

                "team_score":
                    team_score,

                "normal_score":
                    normal_score,

                "swapped_score":
                    swapped_score,

                "time_difference_hours":
                    time_difference,

                "total_score":
                    total_score
            }
        )


    if not candidates:

        return None


    candidates.sort(

        key=lambda item:
            item[
                "total_score"
            ],

        reverse=True
    )


    best = candidates[
        0
    ]


    # Exact / near-exact pair wins safely.

    if best[
        "team_score"
    ] >= 0.96:

        return best


    # Fuzzy match must clearly beat runner-up.

    if len(
        candidates
    ) > 1:

        second = candidates[
            1
        ]


        if (

            best[
                "total_score"
            ]

            -

            second[
                "total_score"
            ]

            <

            AMBIGUOUS_MATCH_MARGIN
        ):

            return None


    return best


# ============================================================
# TEAM-ORIENTED MARKET LINES
# ============================================================

def reference_line_for_team(

    market_row,

    team
):

    home_score = best_similarity(

        team,

        market_row.get(
            "home_team"
        )
    )


    away_score = best_similarity(

        team,

        market_row.get(
            "away_team"
        )
    )


    if max(

        home_score,

        away_score

    ) < 0.60:

        return (
            np.nan,
            np.nan
        )


    if home_score >= away_score:

        return (

            safe_float(
                market_row.get(
                    "home_spread"
                )
            ),

            safe_float(
                market_row.get(
                    "home_spread_price"
                )
            )
        )


    return (

        safe_float(
            market_row.get(
                "away_spread"
            )
        ),

        safe_float(
            market_row.get(
                "away_spread_price"
            )
        )
    )


def hardrock_line_for_team(

    hardrock_row,

    team
):

    p1_score = best_similarity(

        team,

        hardrock_row.get(
            "participant1_name"
        ),

        hardrock_row.get(
            "participant1_short_name"
        ),

        hardrock_row.get(
            "participant1_abbr"
        )
    )


    p2_score = best_similarity(

        team,

        hardrock_row.get(
            "participant2_name"
        ),

        hardrock_row.get(
            "participant2_short_name"
        ),

        hardrock_row.get(
            "participant2_abbr"
        )
    )


    if max(

        p1_score,

        p2_score

    ) < 0.60:

        return (
            np.nan,
            np.nan
        )


    if p1_score >= p2_score:

        return (

            safe_float(
                hardrock_row.get(
                    "participant1_spread"
                )
            ),

            safe_float(
                hardrock_row.get(
                    "participant1_spread_price"
                )
            )
        )


    return (

        safe_float(
            hardrock_row.get(
                "participant2_spread"
            )
        ),

        safe_float(
            hardrock_row.get(
                "participant2_spread_price"
            )
        )
    )


# ============================================================
# WARNINGS
# ============================================================

def build_signal_warnings(
    row
):

    warnings = []


    if safe_int(
        row.get(
            "game_is_actionable"
        )
    ) != 1:

        warnings.append(
            "STARTED_OR_PAST"
        )

        return warnings


    if safe_int(
        row.get(
            "hardrock_event_matched"
        )
    ) != 1:

        warnings.append(
            "HARDROCK_EVENT_UNMATCHED"
        )


    elif safe_int(
        row.get(
            "hardrock_spread_quality_pass"
        )
    ) != 1:

        warnings.append(
            "HARDROCK_HEADLINE_UNAVAILABLE"
        )


    if safe_int(
        row.get(
            "reference_event_matched"
        )
    ) != 1:

        warnings.append(
            "REFERENCE_EVENT_UNMATCHED"
        )


    elif safe_int(
        row.get(
            "reference_market_books"
        )
    ) < REFERENCE_MIN_BOOKS:

        warnings.append(
            "REFERENCE_CONSENSUS_UNAVAILABLE"
        )


    if safe_int(
        row.get(
            "nn_heads_agree"
        )
    ) != 1:

        warnings.append(
            "MODEL_HEADS_DISAGREE"
        )


    if str(
        row.get(
            "nn_input_status",
            ""
        )
    ) == "MISSING_STATIC_CONTEXT":

        warnings.append(
            "MISSING_STATIC_CONTEXT"
        )


    model_market_gap = safe_float(

        row.get(
            "model_vs_consensus_abs_points"
        )
    )


    if pd.notna(
        model_market_gap
    ):

        if (

            model_market_gap

            >=

            OUTLIER_MODEL_MARKET_GAP
        ):

            warnings.append(
                "MODEL_MARKET_OUTLIER"
            )


        elif (

            model_market_gap

            >=

            LARGE_MODEL_MARKET_GAP
        ):

            warnings.append(
                "LARGE_MODEL_MARKET_DISAGREEMENT"
            )


    return warnings


# ============================================================
# STRENGTH
#
# LOW / MEDIUM / HIGH is a composite signal-quality rating.
#
# It is NOT:
#
# - a guarantee
# - a cover probability
# - a lock
# ============================================================

def calculate_strength(
    row
):

    # --------------------------------------------------------
    # Fail closed if we cannot genuinely evaluate the bet.
    # --------------------------------------------------------

    if safe_int(
        row.get(
            "game_is_actionable"
        )
    ) != 1:

        return (
            np.nan,
            "N/A"
        )


    if safe_int(
        row.get(
            "hardrock_available"
        )
    ) != 1:

        return (
            np.nan,
            "N/A"
        )


    if safe_int(
        row.get(
            "hardrock_snapshot_fresh"
        )
    ) != 1:

        return (
            np.nan,
            "N/A"
        )


    if safe_int(
        row.get(
            "reference_market_fresh"
        )
    ) != 1:

        return (
            np.nan,
            "N/A"
        )


    if safe_int(
        row.get(
            "reference_market_books"
        )
    ) < REFERENCE_MIN_BOOKS:

        return (
            np.nan,
            "N/A"
        )


    model_edge = safe_float(

        row.get(
            "model_edge_points"
        )
    )


    execution_advantage = safe_float(

        row.get(
            "hardrock_vs_consensus_points"
        )
    )


    model_market_gap = safe_float(

        row.get(
            "model_vs_consensus_abs_points"
        )
    )


    if (

        pd.isna(
            model_edge
        )

        or

        pd.isna(
            execution_advantage
        )

        or

        pd.isna(
            model_market_gap
        )
    ):

        return (
            np.nan,
            "N/A"
        )


    # --------------------------------------------------------
    # MODEL EDGE
    # Maximum: 35
    # --------------------------------------------------------

    if model_edge <= 0:

        model_edge_score = 0.0


    elif model_edge < 1.0:

        model_edge_score = 6.0


    elif model_edge < 2.0:

        model_edge_score = 16.0


    elif model_edge < 3.0:

        model_edge_score = 24.0


    elif model_edge < 4.5:

        model_edge_score = 30.0


    else:

        model_edge_score = 35.0


    # --------------------------------------------------------
    # HARD ROCK VS CONSENSUS
    # Maximum: 30
    #
    # Positive means Hard Rock gives the model side a more
    # favorable number than the reference median.
    # --------------------------------------------------------

    if execution_advantage <= 0:

        execution_score = 0.0


    elif execution_advantage < 0.5:

        execution_score = 5.0


    elif execution_advantage < 1.0:

        execution_score = 12.0


    elif execution_advantage < 1.5:

        execution_score = 20.0


    elif execution_advantage < 2.5:

        execution_score = 25.0


    else:

        execution_score = 30.0


    # --------------------------------------------------------
    # DATA QUALITY
    # Maximum: 15
    # --------------------------------------------------------

    input_quality = str(

        row.get(
            "nn_input_status",
            ""
        )
    )


    feature_count = safe_int(

        row.get(
            "model_non_null_features"
        )
    )


    if input_quality == "MISSING_STATIC_CONTEXT":

        data_score = 0.0


    elif feature_count == 54:

        data_score = 15.0


    elif feature_count >= 40:

        data_score = 10.0


    else:

        data_score = 4.0


    # --------------------------------------------------------
    # MODEL AGREEMENT
    # Maximum: 15
    # --------------------------------------------------------

    agreement_score = 0.0


    if safe_int(
        row.get(
            "nn_heads_agree"
        )
    ) == 1:

        agreement_score += 5.0


    margin_sd = safe_float(

        row.get(
            "nn_margin_std"
        )
    )


    probability_sd = safe_float(

        row.get(
            "nn_win_probability_std"
        )
    )


    if pd.notna(
        margin_sd
    ):

        if margin_sd <= 1.0:

            agreement_score += 5.0


        elif margin_sd <= 2.0:

            agreement_score += 4.0


        elif margin_sd <= 3.0:

            agreement_score += 2.0


    if pd.notna(
        probability_sd
    ):

        if probability_sd <= 0.03:

            agreement_score += 5.0


        elif probability_sd <= 0.06:

            agreement_score += 3.0


        elif probability_sd <= 0.10:

            agreement_score += 1.0


    # --------------------------------------------------------
    # EXECUTION PRICE QUALITY
    # Maximum: 5
    # --------------------------------------------------------

    hardrock_price = safe_float(

        row.get(
            "hardrock_spread_price"
        )
    )


    if pd.isna(
        hardrock_price
    ):

        price_score = 0.0


    elif (

        hardrock_price >= -120

        and

        hardrock_price <= 120
    ):

        price_score = 5.0


    elif (

        hardrock_price >= -140

        and

        hardrock_price <= 140
    ):

        price_score = 2.0


    else:

        price_score = 0.0


    score = (

        model_edge_score

        +

        execution_score

        +

        data_score

        +

        agreement_score

        +

        price_score
    )


    # ========================================================
    # CONSERVATIVE CAPS
    # ========================================================

    # A tiny model edge should not become MEDIUM/HIGH just
    # because Hard Rock differs from other sportsbooks.

    if (

        model_edge

        <

        MIN_MEANINGFUL_MODEL_EDGE
    ):

        score = min(
            score,
            39.0
        )


    if safe_int(
        row.get(
            "nn_heads_agree"
        )
    ) != 1:

        score = min(
            score,
            39.0
        )


    if input_quality == "MISSING_STATIC_CONTEXT":

        score = min(
            score,
            29.0
        )


    # Early-season imputation cannot be HIGH.

    if input_quality == "EARLY_SEASON_HEAVY_IMPUTATION":

        score = min(
            score,
            69.0
        )


    # Hard Rock not better than consensus:
    # HIGH is not allowed.

    if execution_advantage <= 0:

        score = min(
            score,
            59.0
        )


    # --------------------------------------------------------
    # MODEL / MARKET DISAGREEMENT
    #
    # Massive disagreement is a warning about our model,
    # not automatic evidence that the sportsbooks are wrong.
    # --------------------------------------------------------

    if (

        model_market_gap

        >=

        OUTLIER_MODEL_MARKET_GAP
    ):

        score = min(
            score,
            39.0
        )


    elif (

        model_market_gap

        >=

        LARGE_MODEL_MARKET_GAP
    ):

        score = min(
            score,
            59.0
        )


    if (

        pd.notna(
            margin_sd
        )

        and

        margin_sd > 4.0
    ):

        score = min(
            score,
            59.0
        )


    if (

        pd.notna(
            probability_sd
        )

        and

        probability_sd > 0.10
    ):

        score = min(
            score,
            59.0
        )


    if (

        pd.notna(
            hardrock_price
        )

        and

        hardrock_price < -140
    ):

        score = min(
            score,
            59.0
        )


    score = max(

        0.0,

        min(
            100.0,
            score
        )
    )


    if score >= 70:

        label = "HIGH"


    elif score >= 40:

        label = "MEDIUM"


    else:

        label = "LOW"


    return (
        score,
        label
    )


# ============================================================
# START
# ============================================================

print(
    "BUILD LIVE MODEL + HARD ROCK BOARD — V3"
)


print(
    "=" * 110
)


print(
    "Execution sportsbook: HARD ROCK BET FL"
)


print(
    "Hard Rock source:     ODDSPAPI"
)


print(
    "Reference market:     THE ODDS API MULTI-BOOK CONSENSUS"
)


print(
    "Strength labels:      LOW / MEDIUM / HIGH"
)


print(
    "Outlier rule:         >= 8 pts model-vs-consensus => LOW"
)


print(
    "API calls:            NONE"
)


# ============================================================
# VERIFY INPUTS
# ============================================================

for required_file in [

    DATABASE_FILE,

    PREDICTION_FILE,

    HARDROCK_FILE

]:

    if not required_file.exists():

        raise RuntimeError(

            f"Required file missing:\n"
            f"{required_file}"
        )


# ============================================================
# LOAD MODEL PREDICTIONS
# ============================================================

header(
    "LOAD MODEL PREDICTIONS"
)


predictions = pd.read_csv(
    PREDICTION_FILE
)


predictions[
    "start_date"
] = pd.to_datetime(

    predictions[
        "start_date"
    ],

    utc=True,

    errors="coerce"
)


now_utc = pd.Timestamp.now(
    tz="UTC"
)


predictions[
    "game_is_actionable"
] = (

    predictions[
        "start_date"
    ].notna()

    &

    (
        predictions[
            "start_date"
        ]

        >

        now_utc
    )

).astype(
    int
)


print(

    f"Prediction rows:             "
    f"{len(predictions):,}"
)


print(

    f"Actionable pregame rows:    "
    f"{int(predictions['game_is_actionable'].sum()):,}"
)


print(

    f"Started/past model rows:    "
    f"{int((predictions['game_is_actionable'] == 0).sum()):,}"
)


required_prediction_columns = [

    "game_id",

    "start_date",

    "home_team",

    "away_team",

    "model_favorite",

    "model_spread",

    "model_spread_display",

    "model_favorite_win_probability",

    "nn_margin_std",

    "nn_win_probability_std",

    "nn_heads_agree",

    "nn_input_status",

    "model_non_null_features"
]


missing = [

    column

    for column
    in required_prediction_columns

    if column not in predictions.columns
]


if missing:

    raise RuntimeError(

        "Prediction file missing columns:\n"

        +

        "\n".join(
            missing
        )
    )


# ============================================================
# LOAD HARD ROCK EXECUTION SNAPSHOT
# ============================================================

header(
    "LOAD HARD ROCK EXECUTION SNAPSHOT"
)


hardrock = pd.read_csv(
    HARDROCK_FILE
)


hardrock[
    "captured_at_utc"
] = pd.to_datetime(

    hardrock[
        "captured_at_utc"
    ],

    utc=True,

    errors="coerce"
)


hardrock[
    "start_time_utc"
] = pd.to_datetime(

    hardrock[
        "start_time_utc"
    ],

    utc=True,

    errors="coerce"
)


if len(
    hardrock
) == 0:

    raise RuntimeError(

        "Hard Rock latest CSV "
        "contains zero rows."
    )


required_hardrock_columns = [

    "fixture_id",

    "start_time_utc",

    "status_name",

    "participant1_name",

    "participant1_short_name",

    "participant1_abbr",

    "participant2_name",

    "participant2_short_name",

    "participant2_abbr",

    "participant1_spread",

    "participant1_spread_price",

    "participant2_spread",

    "participant2_spread_price",

    "spread_quality_pass",

    "spread_quality_status",

    "bookmaker_active",

    "suspended"
]


missing_hardrock = [

    column

    for column
    in required_hardrock_columns

    if column not in hardrock.columns
]


if missing_hardrock:

    raise RuntimeError(

        "Hard Rock file missing V2 quality columns:\n"

        +

        "\n".join(
            missing_hardrock
        )

        +

        "\n\nRun "
        "snapshot_hardrock_oddspapi_2026_v1.py first."
    )


hardrock_batch_time = hardrock[
    "captured_at_utc"
].max()


hardrock_age_hours = (

    (
        now_utc
        -
        hardrock_batch_time
    ).total_seconds()

    /

    3600.0
)


hardrock_fresh = int(

    hardrock_age_hours

    <=

    HARDROCK_MAX_SNAPSHOT_AGE_HOURS
)


print(

    f"Hard Rock rows:              "
    f"{len(hardrock):,}"
)


print(

    f"Captured UTC:                "
    f"{hardrock_batch_time.isoformat()}"
)


print(

    f"Snapshot age:                "
    f"{hardrock_age_hours:.2f} hours"
)


print(

    f"Snapshot fresh:              "
    f"{'YES' if hardrock_fresh else 'NO'}"
)


print(

    f"Usable headline rows:       "
    f"{int((hardrock['spread_quality_pass'] == 1).sum()):,}"
)


# ============================================================
# LOAD LATEST REFERENCE MARKET BATCH
# ============================================================

header(
    "LOAD REFERENCE MARKET"
)


connection = sqlite3.connect(
    DATABASE_FILE
)


try:

    latest_batch = pd.read_sql_query(

        """
        SELECT
            batch_id,
            captured_at_utc

        FROM execution_market_snapshots

        WHERE
            sport = ?
            AND is_pregame = 1

        ORDER BY
            captured_at_utc DESC

        LIMIT 1
        """,

        connection,

        params=[
            SPORT
        ]
    )


    if len(
        latest_batch
    ) == 0:

        raise RuntimeError(

            "No pregame NCAAF reference "
            "snapshots found."
        )


    reference_batch_id = str(

        latest_batch.iloc[
            0
        ][
            "batch_id"
        ]
    )


    reference_batch_time = pd.to_datetime(

        latest_batch.iloc[
            0
        ][
            "captured_at_utc"
        ],

        utc=True
    )


    reference_market = pd.read_sql_query(

        """
        SELECT *

        FROM execution_market_snapshots

        WHERE
            batch_id = ?
            AND sport = ?
            AND is_pregame = 1
        """,

        connection,

        params=[

            reference_batch_id,

            SPORT
        ]
    )


finally:

    connection.close()


reference_market[
    "captured_at_utc"
] = pd.to_datetime(

    reference_market[
        "captured_at_utc"
    ],

    utc=True,

    errors="coerce"
)


reference_market[
    "commence_time_utc"
] = pd.to_datetime(

    reference_market[
        "commence_time_utc"
    ],

    utc=True,

    errors="coerce"
)


reference_age_hours = (

    (
        now_utc
        -
        reference_batch_time
    ).total_seconds()

    /

    3600.0
)


reference_fresh = int(

    reference_age_hours

    <=

    REFERENCE_MAX_SNAPSHOT_AGE_HOURS
)


print(

    f"Batch:                       "
    f"{reference_batch_id}"
)


print(

    f"Reference rows:              "
    f"{len(reference_market):,}"
)


print(

    f"Captured UTC:                "
    f"{reference_batch_time.isoformat()}"
)


print(

    f"Snapshot age:                "
    f"{reference_age_hours:.2f} hours"
)


print(

    f"Snapshot fresh:              "
    f"{'YES' if reference_fresh else 'NO'}"
)


# ============================================================
# ONE ROW PER EVENT
# ============================================================

# Only pregame Hard Rock fixtures can generate pregame bets.

hardrock_pregame = hardrock[

    hardrock[
        "status_name"
    ]

    .astype(
        str
    )

    .str.lower()

    .str.contains(
        "pre",
        na=False
    )

].copy()


hardrock_events = hardrock_pregame[

    [
        "fixture_id",

        "start_time_utc",

        "participant1_short_name",

        "participant2_short_name"
    ]

].drop_duplicates(

    subset=[
        "fixture_id"
    ]
)


reference_events = (

    reference_market

    .groupby(
        "odds_api_event_id",
        as_index=False
    )

    .agg(
        {
            "commence_time_utc":
                "first",

            "home_team":
                "first",

            "away_team":
                "first"
        }
    )
)


# ============================================================
# BUILD BOARD
# ============================================================

header(
    "MATCH MODEL GAMES TO HARD ROCK + REFERENCE MARKET"
)


board_rows = []


hardrock_matches = 0

hardrock_usable_games = 0

hardrock_rejected_games = 0

reference_matches = 0

reference_consensus_games = 0


for _, prediction in predictions.iterrows():

    row = prediction.to_dict()


    favorite = str(

        prediction[
            "model_favorite"
        ]
    )


    model_spread = safe_float(

        prediction[
            "model_spread"
        ]
    )


    row.update(

        {
            "market_status":
                (
                    "PREGAME"

                    if safe_int(
                        prediction[
                            "game_is_actionable"
                        ]
                    )
                    ==
                    1

                    else
                    "STARTED_OR_PAST"
                ),

            "hardrock_fixture_id":
                None,

            "hardrock_event_matched":
                0,

            "hardrock_event_match_score":
                np.nan,

            "hardrock_event_time_difference_hours":
                np.nan,

            "hardrock_snapshot_age_hours":
                hardrock_age_hours,

            "hardrock_snapshot_fresh":
                hardrock_fresh,

            "hardrock_spread_quality_pass":
                0,

            "hardrock_spread_quality_status":
                None,

            "hardrock_available":
                0,

            "hardrock_spread":
                np.nan,

            "hardrock_spread_price":
                np.nan,

            "hardrock_display":
                (
                    "Game started/past"

                    if safe_int(
                        prediction[
                            "game_is_actionable"
                        ]
                    )
                    !=
                    1

                    else
                    "Unavailable"
                ),

            "hardrock_raw_candidate_display":
                "N/A",

            "hardrock_fixture_path":
                None,

            "reference_event_id":
                None,

            "reference_event_matched":
                0,

            "reference_event_match_score":
                np.nan,

            "reference_event_time_difference_hours":
                np.nan,

            "reference_market_batch_id":
                reference_batch_id,

            "reference_market_age_hours":
                reference_age_hours,

            "reference_market_fresh":
                reference_fresh,

            "reference_market_spread":
                np.nan,

            "reference_market_books":
                0,

            "reference_market_display":
                "Unavailable",

            "model_edge_points":
                np.nan,

            "model_edge_display":
                "N/A",

            "model_vs_consensus_points":
                np.nan,

            "model_vs_consensus_abs_points":
                np.nan,

            "hardrock_vs_consensus_points":
                np.nan,

            "hardrock_vs_consensus_display":
                "N/A",

            "signal_warning":
                "",

            "strength_score":
                np.nan,

            "strength_label":
                "N/A"
        }
    )


    # --------------------------------------------------------
    # Started games remain in CSV/JSON for audit,
    # but we do not create a new pregame signal.
    # --------------------------------------------------------

    if safe_int(
        prediction[
            "game_is_actionable"
        ]
    ) != 1:

        row[
            "signal_warning"
        ] = "STARTED_OR_PAST"


        board_rows.append(
            row
        )


        continue


    # ========================================================
    # HARD ROCK MATCH
    # ========================================================

    hardrock_match = find_best_event_match(

        prediction,

        hardrock_events,

        event_id_column=
            "fixture_id",

        event_time_column=
            "start_time_utc",

        team1_column=
            "participant1_short_name",

        team2_column=
            "participant2_short_name"
    )


    if hardrock_match is not None:

        hardrock_matches += 1


        fixture_id = hardrock_match[
            "event_id"
        ]


        row[
            "hardrock_fixture_id"
        ] = fixture_id


        row[
            "hardrock_event_matched"
        ] = 1


        row[
            "hardrock_event_match_score"
        ] = hardrock_match[
            "team_score"
        ]


        row[
            "hardrock_event_time_difference_hours"
        ] = hardrock_match[
            "time_difference_hours"
        ]


        hardrock_row = hardrock_pregame[

            hardrock_pregame[
                "fixture_id"
            ]
            ==
            fixture_id

        ].iloc[
            0
        ]


        row[
            "hardrock_fixture_path"
        ] = hardrock_row.get(
            "fixture_path"
        )


        quality_pass = safe_int(

            hardrock_row.get(
                "spread_quality_pass"
            )
        )


        quality_status = str(

            hardrock_row.get(
                "spread_quality_status"
            )
        )


        row[
            "hardrock_spread_quality_pass"
        ] = quality_pass


        row[
            "hardrock_spread_quality_status"
        ] = quality_status


        active = (

            safe_int(
                hardrock_row.get(
                    "bookmaker_active"
                )
            )

            ==
            1
        )


        suspended = (

            safe_int(
                hardrock_row.get(
                    "suspended"
                )
            )

            ==
            1
        )


        (
            raw_spread,
            raw_price

        ) = hardrock_line_for_team(

            hardrock_row,

            favorite
        )


        if pd.notna(
            raw_spread
        ):

            row[
                "hardrock_raw_candidate_display"
            ] = spread_display(

                favorite,

                raw_spread,

                raw_price
            )


        # ----------------------------------------------------
        # Critical V3 rule:
        #
        # Rejected alternate ladder does NOT become the
        # sportsbook headline and creates no MODEL EDGE.
        # ----------------------------------------------------

        if quality_pass != 1:

            hardrock_rejected_games += 1


            row[
                "hardrock_display"
            ] = "Headline unavailable"


        elif (

            active

            and

            not suspended

            and

            hardrock_fresh == 1

            and

            pd.notna(
                raw_spread
            )
        ):

            hardrock_usable_games += 1


            row[
                "hardrock_available"
            ] = 1


            row[
                "hardrock_spread"
            ] = float(
                raw_spread
            )


            row[
                "hardrock_spread_price"
            ] = (

                float(
                    raw_price
                )

                if pd.notna(
                    raw_price
                )

                else np.nan
            )


            row[
                "hardrock_display"
            ] = spread_display(

                favorite,

                raw_spread,

                raw_price
            )


            if pd.notna(
                model_spread
            ):

                model_edge = (

                    float(
                        raw_spread
                    )

                    -

                    float(
                        model_spread
                    )
                )


                row[
                    "model_edge_points"
                ] = model_edge


                row[
                    "model_edge_display"
                ] = edge_display(

                    favorite,

                    model_edge
                )


    # ========================================================
    # REFERENCE MARKET MATCH
    # ========================================================

    reference_match = find_best_event_match(

        prediction,

        reference_events,

        event_id_column=
            "odds_api_event_id",

        event_time_column=
            "commence_time_utc",

        team1_column=
            "home_team",

        team2_column=
            "away_team"
    )


    if reference_match is not None:

        reference_matches += 1


        event_id = reference_match[
            "event_id"
        ]


        row[
            "reference_event_id"
        ] = event_id


        row[
            "reference_event_matched"
        ] = 1


        row[
            "reference_event_match_score"
        ] = reference_match[
            "team_score"
        ]


        row[
            "reference_event_time_difference_hours"
        ] = reference_match[
            "time_difference_hours"
        ]


        event_market = reference_market[

            reference_market[
                "odds_api_event_id"
            ]
            ==
            event_id

        ].copy()


        # One spread per bookmaker.

        reference_spreads_by_book = {}


        for _, book_row in event_market.iterrows():

            if safe_int(

                book_row.get(
                    "is_execution_book"
                )

            ) == 1:

                continue


            (
                spread,
                _price

            ) = reference_line_for_team(

                book_row,

                favorite
            )


            if pd.isna(
                spread
            ):

                continue


            bookmaker_key = str(

                book_row.get(
                    "bookmaker_key"
                )

                or

                book_row.get(
                    "bookmaker_title"
                )

                or

                f"row_{len(reference_spreads_by_book)}"
            )


            reference_spreads_by_book[
                bookmaker_key
            ] = float(
                spread
            )


        reference_spreads = list(

            reference_spreads_by_book.values()
        )


        if reference_spreads:

            reference_consensus_games += 1


            consensus = float(

                np.median(
                    reference_spreads
                )
            )


            row[
                "reference_market_spread"
            ] = consensus


            row[
                "reference_market_books"
            ] = len(
                reference_spreads
            )


            row[
                "reference_market_display"
            ] = spread_display(

                favorite,

                consensus
            )


            if pd.notna(
                model_spread
            ):

                signed_gap = (

                    consensus

                    -

                    float(
                        model_spread
                    )
                )


                row[
                    "model_vs_consensus_points"
                ] = signed_gap


                row[
                    "model_vs_consensus_abs_points"
                ] = abs(
                    signed_gap
                )


    # ========================================================
    # HARD ROCK VS CONSENSUS
    # ========================================================

    hardrock_spread = safe_float(

        row.get(
            "hardrock_spread"
        )
    )


    consensus = safe_float(

        row.get(
            "reference_market_spread"
        )
    )


    if (

        pd.notna(
            hardrock_spread
        )

        and

        pd.notna(
            consensus
        )
    ):

        execution_advantage = (

            hardrock_spread

            -

            consensus
        )


        row[
            "hardrock_vs_consensus_points"
        ] = execution_advantage


        if abs(
            execution_advantage
        ) < 0.05:

            row[
                "hardrock_vs_consensus_display"
            ] = "Same as consensus"


        elif execution_advantage > 0:

            row[
                "hardrock_vs_consensus_display"
            ] = (

                f"{favorite} "
                f"+{execution_advantage:.1f} pts "
                f"vs consensus"
            )


        else:

            row[
                "hardrock_vs_consensus_display"
            ] = (

                f"{favorite} "
                f"{execution_advantage:.1f} pts "
                f"vs consensus"
            )


    # ========================================================
    # WARNINGS + STRENGTH
    # ========================================================

    warnings = build_signal_warnings(
        row
    )


    row[
        "signal_warning"
    ] = (

        " | ".join(
            warnings
        )

        if warnings

        else "NONE"
    )


    (
        strength_score,
        strength_label

    ) = calculate_strength(
        row
    )


    row[
        "strength_score"
    ] = strength_score


    row[
        "strength_label"
    ] = strength_label


    board_rows.append(
        row
    )


# ============================================================
# DATAFRAME
# ============================================================

board = pd.DataFrame(
    board_rows
)


board[
    "start_date"
] = pd.to_datetime(

    board[
        "start_date"
    ],

    utc=True,

    errors="coerce"
)


board = board.sort_values(

    by=[

        "start_date",

        "game_id"
    ]

).reset_index(
    drop=True
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_CSV.parent.mkdir(

    parents=True,

    exist_ok=True
)


board.to_csv(

    OUTPUT_CSV,

    index=False
)


board.to_json(

    OUTPUT_JSON,

    orient="records",

    indent=2,

    date_format="iso"
)


# ============================================================
# SUMMARY
# ============================================================

header(
    "BOARD STATUS"
)


print(

    f"Model games loaded:             "
    f"{len(predictions):,}"
)


print(

    f"Actionable pregame games:       "
    f"{int((board['game_is_actionable'] == 1).sum()):,}"
)


print(

    f"Started/past games retained:    "
    f"{int((board['game_is_actionable'] == 0).sum()):,}"
)


print(

    f"Hard Rock event matches:        "
    f"{hardrock_matches:,}"
)


print(

    f"Hard Rock usable headlines:     "
    f"{hardrock_usable_games:,}"
)


print(

    f"Hard Rock rejected headlines:   "
    f"{hardrock_rejected_games:,}"
)


print(

    f"Reference event matches:        "
    f"{reference_matches:,}"
)


print(

    f"Reference consensus games:      "
    f"{reference_consensus_games:,}"
)


print()

print(
    "STRENGTH — ACTIONABLE GAMES ONLY"
)


print(
    "-" * 110
)


actionable_board = board[

    board[
        "game_is_actionable"
    ]
    ==
    1

].copy()


strength_counts = (

    actionable_board[
        "strength_label"
    ]

    .value_counts()

    .reindex(

        [

            "HIGH",

            "MEDIUM",

            "LOW",

            "N/A"
        ],

        fill_value=0
    )
)


for (
    label,
    count

) in strength_counts.items():

    print(

        f"{label:10} "
        f"{int(count):,}"
    )


# ============================================================
# MATCH / MARKET QUALITY DIAGNOSTICS
# ============================================================

problem_rows = actionable_board[

    (
        actionable_board[
            "hardrock_event_matched"
        ]
        !=
        1
    )

    |

    (
        actionable_board[
            "reference_event_matched"
        ]
        !=
        1
    )

    |

    (
        actionable_board[
            "hardrock_spread_quality_pass"
        ]
        !=
        1
    )

].copy()


if len(
    problem_rows
) > 0:

    header(
        "MATCH / MARKET QUALITY DIAGNOSTICS"
    )


    for _, row in problem_rows.iterrows():

        print(

            f"{row['away_team']} "
            f"@ "
            f"{row['home_team']}"
        )


        print(

            f"  Hard Rock match:   "
            f"{'YES' if safe_int(row['hardrock_event_matched']) == 1 else 'NO'}"
        )


        print(

            f"  HR quality:        "
            f"{row.get('hardrock_spread_quality_status')}"
        )


        print(

            f"  Reference match:   "
            f"{'YES' if safe_int(row['reference_event_matched']) == 1 else 'NO'}"
        )


        print(

            f"  Reference books:   "
            f"{safe_int(row['reference_market_books'])}"
        )


        print(

            f"  Warning:           "
            f"{row['signal_warning']}"
        )


        print(
            "-" * 110
        )


# ============================================================
# LIVE USER-FACING BOARD
# ============================================================

header(
    "LIVE BETTING BOARD — ACTIONABLE GAMES"
)


for _, row in actionable_board.iterrows():

    probability = safe_float(

        row[
            "model_favorite_win_probability"
        ]
    )


    print(

        f"{row['away_team']} "
        f"@ "
        f"{row['home_team']}"
    )


    print()


    print(
        "MODEL SPREAD"
    )


    print(

        f"  "
        f"{row['model_spread_display']}"
    )


    print()


    print(
        "HARD ROCK"
    )


    print(

        f"  "
        f"{row['hardrock_display']}"
    )


    print()


    print(
        "MODEL EDGE"
    )


    print(

        f"  "
        f"{row['model_edge_display']}"
    )


    print()


    print(
        "WIN PROBABILITY"
    )


    if pd.notna(
        probability
    ):

        print(

            f"  "
            f"{probability * 100:.1f}%"
        )


    else:

        print(
            "  N/A"
        )


    print()


    print(
        "STRENGTH"
    )


    print(

        f"  "
        f"{row['strength_label']}"
    )


    if row[
        "signal_warning"
    ] != "NONE":

        print()


        print(
            "WARNING"
        )


        print(

            f"  "
            f"{row['signal_warning']}"
        )


    print()


    print(
        "REFERENCE MARKET"
    )


    print(

        f"  "
        f"{row['reference_market_display']}"

        f" | "

        f"{safe_int(row['reference_market_books'])} books"
    )


    print()


    print(
        "HARD ROCK VS CONSENSUS"
    )


    print(

        f"  "
        f"{row['hardrock_vs_consensus_display']}"
    )


    model_market_gap = safe_float(

        row.get(
            "model_vs_consensus_abs_points"
        )
    )


    if pd.notna(
        model_market_gap
    ):

        print()


        print(
            "MODEL VS CONSENSUS GAP"
        )


        print(

            f"  "
            f"{model_market_gap:.1f} pts"
        )


    print()


    print(
        "DIAGNOSTICS"
    )


    print(

        f"  Input: "
        f"{row['nn_input_status']}"
    )


    print(

        f"  Ensemble margin SD: "
        f"{safe_float(row['nn_margin_std']):.2f}"
    )


    print(

        f"  Win probability SD: "
        f"{safe_float(row['nn_win_probability_std']):.3f}"
    )


    print(

        f"  Heads agree: "
        f"{'YES' if safe_int(row['nn_heads_agree']) == 1 else 'NO'}"
    )


    if pd.notna(
        row.get(
            "strength_score"
        )
    ):

        print(

            f"  Strength score: "
            f"{float(row['strength_score']):.1f}/100"
        )


    else:

        print(
            "  Strength score: N/A"
        )


    print(
        "-" * 110
    )


# ============================================================
# COMPLETE
# ============================================================

header(
    "MODEL + HARD ROCK BOARD COMPLETE"
)


print(
    "Files written:"
)


print(

    f"  "
    f"{OUTPUT_CSV}"
)


print(

    f"  "
    f"{OUTPUT_JSON}"
)


print()


print(
    "RULES NOW ENFORCED:"
)


print(
    "  Hard Rock spread_quality_pass must equal 1."
)


print(
    "  Rejected Hard Rock ladders never generate MODEL EDGE."
)


print(
    "  Model edge below 1 point cannot exceed LOW."
)


print(
    "  6+ point model/consensus gap cannot exceed MEDIUM."
)


print(
    "  8+ point model/consensus gap is forced LOW."
)


print(
    "  Started/past games are not actionable."
)


print()


print(
    "Strength remains prospectively unvalidated "
    "and is not a cover probability."
)