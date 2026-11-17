"""
01_download_data.py
Download Eurovision voting and contestant datasets.
- Spijkervet Eurovision Dataset (1956-2023): votes.csv, contestants.csv
- josago97/EurovisionDataset (1957-2024): structured dataset exported to votes_josago.csv
"""
import os
import requests
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
os.makedirs(DATA_DIR, exist_ok=True)

SPIJKERVET_BASE = "https://github.com/Spijkervet/eurovision-dataset/releases/download/2023"
JOSAGO_JSON_URL = "https://raw.githubusercontent.com/josago97/EurovisionDataset/main/dataset/eurovision.json"

def download_file(url: str, dest_path: str):
    print(f"Downloading {url} -> {dest_path}...")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    resp = requests.get(url, headers=headers, stream=True, timeout=60)
    resp.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print(f"Successfully saved {dest_path} ({os.path.getsize(dest_path)} bytes)")

def download_and_convert_josago(dest_path: str):
    print(f"Fetching josago97 dataset from {JOSAGO_JSON_URL}...")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    resp = requests.get(JOSAGO_JSON_URL, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    
    rows = []
    for contest in data:
        year = contest.get("year")
        contestants_map = {c["id"]: c.get("country", "") for c in contest.get("contestants", [])}
        for rd in contest.get("rounds", []):
            round_name = rd.get("name")
            for perf in rd.get("performances", []):
                to_cid = perf.get("contestantId")
                to_country = contestants_map.get(to_cid, "")
                scores_by_type = {}
                for sc in perf.get("scores", []):
                    scores_by_type[sc.get("name")] = sc.get("votes", {})
                
                total_votes = scores_by_type.get("total", {})
                tele_votes = scores_by_type.get("public", {})
                jury_votes = scores_by_type.get("jury", {})
                
                all_voters = set(total_votes.keys()) | set(tele_votes.keys()) | set(jury_votes.keys())
                for voter in all_voters:
                    rows.append({
                        "year": year,
                        "round": round_name,
                        "from_country_id": voter.lower(),
                        "to_country_id": to_country.lower(),
                        "from_country": voter,
                        "to_country": to_country,
                        "total_points": total_votes.get(voter, 0),
                        "tele_points": tele_votes.get(voter, None),
                        "jury_points": jury_votes.get(voter, None)
                    })
    df = pd.DataFrame(rows)
    df.to_csv(dest_path, index=False)
    print(f"Successfully converted and saved josago97 votes -> {dest_path} ({len(df)} rows)")

def main():
    # 1. Spijkervet votes.csv
    dest_votes_spijkervet = os.path.join(DATA_DIR, "votes_spijkervet.csv")
    if not os.path.exists(dest_votes_spijkervet):
        download_file(f"{SPIJKERVET_BASE}/votes.csv", dest_votes_spijkervet)
    else:
        print(f"Already exists: {dest_votes_spijkervet}")

    # 2. Spijkervet contestants.csv
    dest_contestants_spijkervet = os.path.join(DATA_DIR, "contestants_spijkervet.csv")
    if not os.path.exists(dest_contestants_spijkervet):
        download_file(f"{SPIJKERVET_BASE}/contestants.csv", dest_contestants_spijkervet)
    else:
        print(f"Already exists: {dest_contestants_spijkervet}")

    # 3. josago97 votes.csv
    dest_votes_josago = os.path.join(DATA_DIR, "votes_josago.csv")
    if not os.path.exists(dest_votes_josago):
        download_and_convert_josago(dest_votes_josago)
    else:
        print(f"Already exists: {dest_votes_josago}")

    # 4. CEPII dist_cepii.dta
    dest_dist_cepii = os.path.join(DATA_DIR, "dist_cepii.dta")
    if not os.path.exists(dest_dist_cepii):
        download_file("http://www.cepii.fr/distance/dist_cepii.dta", dest_dist_cepii)
    else:
        print(f"Already exists: {dest_dist_cepii}")

if __name__ == "__main__":
    main()
