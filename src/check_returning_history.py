import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("CFBD_API_KEY")

if not API_KEY:
    raise RuntimeError("CFBD_API_KEY was not found")

url = "https://api.collegefootballdata.com/player/returning"

headers = {
    "Authorization": f"Bearer {API_KEY}"
}

years = range(2018, 2026)

fields_to_check = [
    "totalPPA",
    "totalPassingPPA",
    "totalReceivingPPA",
    "totalRushingPPA",
    "percentPPA",
    "percentPassingPPA",
    "percentReceivingPPA",
    "percentRushingPPA",
    "usage",
    "passingUsage",
    "receivingUsage",
    "rushingUsage"
]

print("RETURNING PRODUCTION HISTORY AUDIT")
print("=" * 90)

for year in years:

    response = requests.get(
        url,
        headers=headers,
        params={
            "year": year
        },
        timeout=60
    )

    print(f"\n{year}")
    print("-" * 90)

    if response.status_code != 200:

        print(
            f"Request failed with status "
            f"{response.status_code}"
        )

        continue

    records = response.json()

    print(f"Teams returned: {len(records)}")

    if not records:
        continue

    # --------------------------------------------------
    # FIELD COVERAGE
    # --------------------------------------------------

    for field in fields_to_check:

        count = sum(
            1
            for record in records
            if record.get(field) is not None
        )

        coverage = (
            count /
            len(records)
        ) * 100

        print(
            f"{field:<25}"
            f"{count:>4} / {len(records):<4}"
            f"{coverage:>7.2f}%"
        )


    # --------------------------------------------------
    # ALABAMA SPOT CHECK
    # --------------------------------------------------

    alabama = next(
        (
            record
            for record in records
            if record.get("team") == "Alabama"
        ),
        None
    )

    if alabama:

        print("\nAlabama spot check:")

        print(
            f"  Overall PPA returning: "
            f"{alabama.get('percentPPA')}"
        )

        print(
            f"  Passing PPA returning: "
            f"{alabama.get('percentPassingPPA')}"
        )

        print(
            f"  Receiving PPA returning: "
            f"{alabama.get('percentReceivingPPA')}"
        )

        print(
            f"  Rushing PPA returning: "
            f"{alabama.get('percentRushingPPA')}"
        )

    else:

        print("\nAlabama not present.")