import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SRC_DIRECTORY = PROJECT_ROOT / "src"

DATA_DIRECTORY = PROJECT_ROOT / "data"

DATABASE_FILE = DATA_DIRECTORY / "cfb.db"

LOG_DIRECTORY = DATA_DIRECTORY / "logs"

PROCESSED_DIRECTORY = DATA_DIRECTORY / "processed"

STATUS_FILE = (
    PROCESSED_DIRECTORY
    / "game_day_status_v1.json"
)


# ============================================================
# ACTIVE GAME-DAY PIPELINE
# ============================================================

PIPELINE_STEPS = [

    {
        "name": "Collect Market",

        "display": (
            "COLLECTING SPORTSBOOK DATA"
        ),

        "script": (
            "snapshot_execution_market_2026_v1.py"
        ),

        "progress": 25
    },

    {
        "name": "Freeze Signals",

        "display": (
            "ANALYZING MARKET SIGNALS"
        ),

        "script": (
            "freeze_live_market_signals_v1.py"
        ),

        "progress": 55
    },

    {
        "name": "Build Board",

        "display": (
            "BUILDING GAME DAY BOARD"
        ),

        "script": (
            "build_game_day_board_v1.py"
        ),

        "progress": 80
    }

]


# ============================================================
# HELPERS
# ============================================================

def utc_now():

    return datetime.now(
        timezone.utc
    ).replace(
        microsecond=0
    ).isoformat()


def write_status(
    state,
    message,
    progress,
    current_step=None,
    error=None
):

    PROCESSED_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )


    status = {

        "updated_at_utc":
            utc_now(),

        "state":
            state,

        "message":
            message,

        "progress":
            progress,

        "current_step":
            current_step,

        "error":
            error
    }


    with open(
        STATUS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            status,
            file,
            indent=2
        )


def print_header(
    text
):

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


def check_file(
    path,
    description
):

    if not path.exists():

        raise RuntimeError(

            f"{description} was not found:\n"
            f"{path}"
        )


def run_script(
    script_name,
    log_file
):

    script_path = (
        SRC_DIRECTORY
        /
        script_name
    )


    check_file(
        script_path,
        f"Pipeline script {script_name}"
    )


    print()

    print(
        f"Running: {script_name}"
    )

    print(
        "-" * 110
    )


    process = subprocess.Popen(

        [
            sys.executable,
            str(
                script_path
            )
        ],

        cwd=PROJECT_ROOT,

        stdout=subprocess.PIPE,

        stderr=subprocess.STDOUT,

        text=True,

        bufsize=1
    )


    output_lines = []


    assert process.stdout is not None


    for line in process.stdout:

        print(
            line,
            end=""
        )


        output_lines.append(
            line
        )


        with open(
            log_file,
            "a",
            encoding="utf-8"
        ) as log:

            log.write(
                line
            )


    return_code = process.wait()


    if return_code != 0:

        raise RuntimeError(

            f"{script_name} failed "
            f"with exit code {return_code}."
        )


    return "".join(
        output_lines
    )


# ============================================================
# SYSTEM CHECK
# ============================================================

def system_checks():

    print_header(
        "SYSTEM CHECK"
    )


    load_dotenv(
        PROJECT_ROOT
        /
        ".env"
    )


    required_environment_variables = [

        "CFBD_API_KEY",

        "THE_ODDS_API_KEY"
    ]


    failures = []


    for variable in required_environment_variables:

        value = os.getenv(
            variable
        )


        if value:

            print(
                f"✓ {variable}"
            )


        else:

            print(
                f"✗ {variable}"
            )


            failures.append(
                variable
            )


    if DATABASE_FILE.exists():

        print(
            f"✓ Database: "
            f"{DATABASE_FILE}"
        )


    else:

        print(
            f"✗ Database missing: "
            f"{DATABASE_FILE}"
        )


        failures.append(
            "database"
        )


    for step in PIPELINE_STEPS:

        script_path = (

            SRC_DIRECTORY
            /
            step[
                "script"
            ]
        )


        if script_path.exists():

            print(
                f"✓ {step['script']}"
            )


        else:

            print(
                f"✗ {step['script']}"
            )


            failures.append(
                step[
                    "script"
                ]
            )


    if failures:

        raise RuntimeError(

            "System check failed:\n"
            +
            "\n".join(
                failures
            )
        )


    print()

    print(
        "SYSTEM READY"
    )


# ============================================================
# PIPELINE
# ============================================================

def run_pipeline(
    dry_run=False
):

    started_at = datetime.now(
        timezone.utc
    )


    batch_name = started_at.strftime(
        "%Y%m%dT%H%M%SZ"
    )


    LOG_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )


    log_file = (

        LOG_DIRECTORY
        /
        f"game_day_{batch_name}.log"
    )


    print_header(
        "CFB EDGE ENGINE — GAME DAY V2"
    )


    print(
        f"Started UTC: {utc_now()}"
    )


    print(
        f"Project:     {PROJECT_ROOT}"
    )


    print(
        f"Log file:    {log_file}"
    )


    write_status(

        state="STARTING",

        message=(
            "Initializing game-day system"
        ),

        progress=0
    )


    try:

        # ====================================================
        # SYSTEM CHECK
        # ====================================================

        write_status(

            state="RUNNING",

            message=(
                "Running system checks"
            ),

            progress=5,

            current_step=(
                "System Check"
            )
        )


        system_checks()


        write_status(

            state="RUNNING",

            message="System ready",

            progress=10,

            current_step=(
                "System Check"
            )
        )


        # ====================================================
        # DRY RUN
        # ====================================================

        if dry_run:

            print_header(
                "DRY RUN"
            )


            print(
                "System checks passed."
            )


            print()

            print(
                "No API requests were made."
            )


            print()

            print(
                "Pipeline steps that WOULD run:"
            )


            for step in PIPELINE_STEPS:

                print(
                    f"  • "
                    f"{step['display']}"
                )


            write_status(

                state="COMPLETE",

                message=(
                    "Dry run complete"
                ),

                progress=100,

                current_step="Complete"
            )


            return


        # ====================================================
        # RUN ACTIVE PIPELINE
        # ====================================================

        for step in PIPELINE_STEPS:

            write_status(

                state="RUNNING",

                message=step[
                    "display"
                ],

                progress=step[
                    "progress"
                ],

                current_step=step[
                    "name"
                ]
            )


            print_header(
                step[
                    "display"
                ]
            )


            run_script(

                step[
                    "script"
                ],

                log_file
            )


        # ====================================================
        # COMPLETE
        # ====================================================

        finished_at = datetime.now(
            timezone.utc
        )


        elapsed_seconds = (

            finished_at
            -
            started_at

        ).total_seconds()


        write_status(

            state="COMPLETE",

            message=(
                "Game-day pipeline complete"
            ),

            progress=100,

            current_step="Complete"
        )


        print_header(
            "GAME DAY COMPLETE"
        )


        print(
            f"Finished UTC: "
            f"{utc_now()}"
        )


        print(
            f"Runtime:      "
            f"{elapsed_seconds:.1f} seconds"
        )


        print()

        print(
            "✓ Market snapshot updated"
        )


        print(
            "✓ Prospective signals checked"
        )


        print(
            "✓ Game-day board rebuilt"
        )


        print()

        print(
            f"Status file:"
        )


        print(
            STATUS_FILE
        )


        print()

        print(
            f"Log file:"
        )


        print(
            log_file
        )


        print()

        print(
            "Current mode: RESEARCH"
        )


        print(
            "Strength scoring is "
            "PENDING VALIDATION."
        )


        print(
            "Hard Rock NCAAF prices are "
            "required before signals can "
            "be treated as executable."
        )


    except Exception as error:

        write_status(

            state="FAILED",

            message=(
                "Game-day pipeline failed"
            ),

            progress=0,

            current_step="Error",

            error=str(
                error
            )
        )


        print_header(
            "PIPELINE FAILED"
        )


        print(
            str(
                error
            )
        )


        print()

        print(
            "Status written to:"
        )


        print(
            STATUS_FILE
        )


        raise


# ============================================================
# COMMAND LINE
# ============================================================

def main():

    parser = argparse.ArgumentParser(

        description=(

            "Run the CFB Edge Engine "
            "game-day pipeline."
        )
    )


    parser.add_argument(

        "--dry-run",

        action="store_true",

        help=(

            "Check system without "
            "making API requests."
        )
    )


    arguments = parser.parse_args()


    run_pipeline(
        dry_run=arguments.dry_run
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()