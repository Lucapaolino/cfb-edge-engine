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

# Start with one team and one season
params = {
    "year": 2025,
    "team": "Alabama"
}

response = requests.get(
    url,
    headers=headers,
    params=params,
    timeout=60
)

print("Status:", response.status_code)

response.raise_for_status()

records = response.json()

print("Records returned:", len(records))

if records:

    print("\nAvailable fields:")
    print(records[0].keys())

    print("\nAlabama 2025 returning production:")
    print("-" * 80)

    print(records[0])

    print("\nReadable breakdown:")
    print("-" * 80)

    record = records[0]

    for key, value in record.items():
        print(f"{key:<25}: {value}")