import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("CFBD_API_KEY")

if not API_KEY:
    raise RuntimeError("CFBD_API_KEY was not found")

url = "https://api.collegefootballdata.com/talent"

headers = {
    "Authorization": f"Bearer {API_KEY}"
}

params = {
    "year": 2025
}

response = requests.get(
    url,
    headers=headers,
    params=params,
    timeout=60
)

print("Status:", response.status_code)

response.raise_for_status()

teams = response.json()

print("Teams returned:", len(teams))

if teams:

    print("\nAvailable fields:")
    print(teams[0].keys())

    print("\nFirst 10 records:")
    print("-" * 60)

    for team in teams[:10]:
        print(team)

    print("\nAlabama:")
    print("-" * 60)

    for team in teams:

        if team.get("team") == "Alabama":
            print(team)