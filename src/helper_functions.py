"""Helper functions shared by the project notebooks."""

import pandas as pd
import requests

POINT_COLS = ["total_points", "jury_points", "tele_points"]


def download_and_convert_josago(url: str, dest_path: str):
    """Download the EurovisionDataset JSON and save it as a flat vote table (one row per voter -> performer)."""
    print(f"Fetching josago97 dataset from {url}...")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    resp = requests.get(url, headers=headers, timeout=60)
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

                # one row per voter that appears in any of the three score types
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


def missing_table(votes: pd.DataFrame, period: pd.Series) -> pd.DataFrame:
    """Number of rows and of missing values per point column, by period and round."""
    keys = [period.rename("period"), votes["round"]]
    table = votes[POINT_COLS].isna().groupby(keys, observed=True).sum()
    table.insert(0, "rows", votes.groupby(keys, observed=True).size())
    return table


def missing_jury_voters(votes: pd.DataFrame) -> pd.Series:
    """Grand Final voters that have televote points but no jury points."""
    mask = (votes["round"] == "final") & votes["tele_points"].notna() & votes["jury_points"].isna()
    return votes[mask].groupby(["year", "from_country"]).size()


def zero_share(votes: pd.DataFrame, period: pd.Series) -> pd.Series:
    """Share of Grand Final rows with total_points = 0, by period."""
    final = votes["round"] == "final"
    return (votes.loc[final, "total_points"] == 0).groupby(period[final].rename("period"), observed=True).mean()
