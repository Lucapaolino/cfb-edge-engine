import os
import requests
from dotenv import load_dotenv

# Load variables from our .env file
load_dotenv()

# Read the CFBD API key
API_KEY = os.getenv("CFBD_API_KEY")

# Make sure we actually found the key
if not API_KEY:
    raise RuntimeError("CFBD_API_KEY was not found")

# CFBD games endpoint
url = "https://api.collegefootballdata.com/games"

# Authentication
headers = {
    "Authorization": f"Bearer {API_KEY}"
}

# Data we are requesting
params = {
    "year": 2025,
    "seasonType": "regular",
    "classification": "fbs"
}

# Send request to CFBD
response = requests.get(
    url,
    headers=headers,
    params=params,
    timeout=30
)

print("Status:", response.status_code)

# Stop if CFBD returned an error
response.raise_for_status()

# Convert JSON response into Python
games = response.json()

print("Games returned:", len(games))

# For now, inspect what CFBD actually gave us
if games:
    print("\nFirst game's available fields:")
    print(games[0].keys())

    print("\nFirst game:")
    print(games[0])