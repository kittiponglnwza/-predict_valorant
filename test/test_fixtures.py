import requests

API_KEY = "745c5b802b204590bfa05c093f00bd43"
headers = {"X-Auth-Token": API_KEY}

r = requests.get(
    "https://api.football-data.org/v4/competitions/PL/matches",
    headers=headers,
    params={"status": "SCHEDULED"},
    timeout=8
)

print("Status:", r.status_code)

matches = sorted(r.json().get("matches", []), key=lambda x: x["utcDate"])[:5]
for m in matches:
    print(m["utcDate"], m["homeTeam"]["shortName"], "vs", m["awayTeam"]["shortName"])
