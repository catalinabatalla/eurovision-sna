"""
02_build_networks.py
Construct longitudinal directed weighted and binarized networks for Eurovision voting blocs (1975-2025).

Eras:
- P1 (1975–1996): Jury only (12-10-8... introduced in 1975)
- P2 (1997–2008): Televoting introduced; semi-finals introduced in 2004
- P3 (2009–2015): 50% jury + 50% televote combined
- P4 (2016–2025): Combined total points used for cross-era comparability

Network Construction:
- Directed weighted network per period.
- For each year, draw binary directed edge A -> B if B is in A's top-k (default k=3: 8, 10, 12 points).
- Edge weight W(A->B) = (years A put B in top-k) / (years A voted and B competed in Grand Final).
- Binarized networks:
    - Main: weight >= 0.5 -> data/processed/{period}_adj_binary_t05.csv
    - Robustness: weight >= 0.33 -> data/processed/robustness/{period}_adj_binary_t033.csv
    - Robustness k=2, k=4: exported to data/processed/robustness/
- CEPII dyadic attributes: aligned contig and comlang_off adjacency matrices.
"""
import os
import numpy as np
import pandas as pd
import pyreadstat

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(BASE_DIR, "data", "raw")
DATA_PROC = os.path.join(BASE_DIR, "data", "processed")
DATA_ROBUST = os.path.join(DATA_PROC, "robustness")

os.makedirs(DATA_PROC, exist_ok=True)
os.makedirs(DATA_ROBUST, exist_ok=True)

ERAS = {
    "P1": (1975, 1996),
    "P2": (1997, 2008),
    "P3": (2009, 2015),
    "P4": (2016, 2024),
}

ISO2_TO_ISO3 = {
    "AL": "ALB", "AD": "AND", "AM": "ARM", "AU": "AUS", "AT": "AUT", "AZ": "AZE",
    "BY": "BLR", "BE": "BEL", "BA": "BIH", "BG": "BGR", "HR": "HRV", "CY": "CYP",
    "CZ": "CZE", "DK": "DNK", "EE": "EST", "FI": "FIN", "FR": "FRA", "GE": "GEO",
    "DE": "DEU", "GR": "GRC", "HU": "HUN", "IS": "ISL", "IE": "IRL", "IL": "ISR",
    "IT": "ITA", "LV": "LVA", "LT": "LTU", "LU": "LUX", "MT": "MLT", "MD": "MDA",
    "MC": "MCO", "ME": "MNE", "MA": "MAR", "NL": "NLD", "MK": "MKD", "NO": "NOR",
    "PL": "POL", "PT": "PRT", "RO": "ROU", "RU": "RUS", "SM": "SMR", "RS": "SRB",
    "CS": "SCG", "SK": "SVK", "SI": "SVN", "ES": "ESP", "SE": "SWE", "CH": "CHE",
    "TR": "TUR", "UA": "UKR", "GB": "GBR", "UK": "GBR", "YU": "YUG"
}

def load_and_merge_votes():
    """Load Spijkervet (1975-2023) and josago97 (2024-2025) votes, filtering Grand Final only."""
    spijkervet_path = os.path.join(DATA_RAW, "votes_spijkervet.csv")
    josago_path = os.path.join(DATA_RAW, "votes_josago.csv")

    df_spijk = pd.read_csv(spijkervet_path)
    df_josago = pd.read_csv(josago_path)

    # Filter Grand Final votes only (1975 to 2023 for Spijkervet)
    df_spijk_final = df_spijk[(df_spijk["year"] >= 1975) & (df_spijk["year"] <= 2023) & (df_spijk["round"] == "final")].copy()

    # Filter Grand Final votes for 2024-2025 from josago
    df_josago_final = df_josago[(df_josago["year"] >= 2024) & (df_josago["round"] == "final")].copy()

    cols = ["year", "from_country_id", "to_country_id", "total_points"]
    df_all = pd.concat([df_spijk_final[cols], df_josago_final[cols]], ignore_index=True)

    # Standardize country codes to upper
    df_all["from_country_id"] = df_all["from_country_id"].astype(str).str.strip().str.upper()
    df_all["to_country_id"] = df_all["to_country_id"].astype(str).str.strip().str.upper()

    # Filter out non-country aggregates (e.g., 'Rest of the World' televote WLD/ROW)
    df_all = df_all[~df_all["from_country_id"].isin(["WLD", "ROW"])]
    df_all = df_all[~df_all["to_country_id"].isin(["WLD", "ROW"])]

    # Map to ISO-3
    df_all["from_iso3"] = df_all["from_country_id"].map(ISO2_TO_ISO3)
    df_all["to_iso3"] = df_all["to_country_id"].map(ISO2_TO_ISO3)

    # Verify no unmapped countries
    unmapped_from = df_all[df_all["from_iso3"].isna()]["from_country_id"].unique()
    unmapped_to = df_all[df_all["to_iso3"].isna()]["to_country_id"].unique()
    if len(unmapped_from) > 0 or len(unmapped_to) > 0:
        raise ValueError(f"Unmapped country codes: from={unmapped_from}, to={unmapped_to}")

    df_all["total_points"] = pd.to_numeric(df_all["total_points"], errors="coerce").fillna(0)
    return df_all

def compute_period_network(df_votes, period_name, year_min, year_max, k=3):
    """
    Compute directed weighted network for a given period and k.
    Weight W(A->B) = (years A put B in top-k) / (years A voted and B competed in final).
    Returns (adj_df, edgelist_df, nodes, topk_years_matrix, co_years_matrix).
    """
    df_p = df_votes[(df_votes["year"] >= year_min) & (df_votes["year"] <= year_max)].copy()

    # Determine all participating nodes in this period
    voters_in_p = set(df_p["from_iso3"].unique())
    performers_in_p = set(df_p["to_iso3"].unique())
    nodes = sorted(list(voters_in_p | performers_in_p))

    # Precompute per-year voters and competitors
    year_voters = {}
    year_performers = {}
    year_topk_edges = set() # (year, from_iso3, to_iso3)

    for yr, yr_group in df_p.groupby("year"):
        voters = set(yr_group["from_iso3"].unique())
        performers = set(yr_group["to_iso3"].unique())
        year_voters[yr] = voters
        year_performers[yr] = performers

        # For each voter, find the top-k receivers
        for voter, vgroup in yr_group.groupby("from_iso3"):
            # Sort performers by points received descending
            # In case of ties, include candidates with points > 0 matching the k-th rank threshold
            sorted_v = vgroup.sort_values(by="total_points", ascending=False)
            pts = sorted_v["total_points"].values
            # Filter to positive points
            pos_mask = sorted_v["total_points"] > 0
            if not pos_mask.any():
                continue
            
            # Determine threshold
            if len(pts) >= k:
                cutoff = pts[k-1]
                if cutoff > 0:
                    topk_targets = sorted_v[sorted_v["total_points"] >= cutoff]["to_iso3"].tolist()
                else:
                    topk_targets = sorted_v[sorted_v["total_points"] > 0]["to_iso3"].tolist()
            else:
                topk_targets = sorted_v[sorted_v["total_points"] > 0]["to_iso3"].tolist()

            for target in topk_targets:
                if target != voter:
                    year_topk_edges.add((yr, voter, target))

    # Calculate period weights W(A->B)
    adj_matrix = pd.DataFrame(0.0, index=nodes, columns=nodes)
    mat_topk_years = pd.DataFrame(0, index=nodes, columns=nodes)
    mat_co_years = pd.DataFrame(0, index=nodes, columns=nodes)
    edge_records = []

    years = sorted(list(year_voters.keys()))
    for u in nodes:
        for v in nodes:
            if u == v:
                continue
            # Years u voted and v competed in Grand Final
            co_years = sum(1 for yr in years if (u in year_voters[yr] and v in year_performers[yr]))
            mat_co_years.loc[u, v] = co_years
            if co_years > 0:
                topk_years = sum(1 for yr in years if (yr, u, v) in year_topk_edges)
                mat_topk_years.loc[u, v] = topk_years
                weight = topk_years / co_years
                adj_matrix.loc[u, v] = weight
                if weight > 0:
                    edge_records.append({
                        "from": u,
                        "to": v,
                        "weight": round(weight, 5),
                        "topk_years": topk_years,
                        "co_participation_years": co_years
                    })

    edgelist_df = pd.DataFrame(edge_records)
    if not edgelist_df.empty:
        edgelist_df = edgelist_df.sort_values(by="weight", ascending=False).reset_index(drop=True)
    return adj_matrix, edgelist_df, nodes, mat_topk_years, mat_co_years

def compute_period_rank_network(df_votes, period_name, year_min, year_max):
    """
    Compute rank-based robustness directed weighted network for a given period.
    Within each year, voter i's points to finalist j are converted to normalized ordinal
    rank scores in [0, 1]:
      1st place (12 pts) -> 1.0, 2nd (10 pts) -> 0.9, 3rd (8 pts) -> 0.8, down to 10th (1 pt) -> 0.1; 0 otherwise.
    In case of ties (e.g. combined points in P4), average rank points are used.
    Averaged across co-participation years:
      W_rank(A->B) = (sum of annual rank scores A gave B) / (years A voted and B competed in final).
    Returns adj_matrix (DataFrame).
    """
    df_p = df_votes[(df_votes["year"] >= year_min) & (df_votes["year"] <= year_max)].copy()

    voters_in_p = set(df_p["from_iso3"].unique())
    performers_in_p = set(df_p["to_iso3"].unique())
    nodes = sorted(list(voters_in_p | performers_in_p))

    # Standard Eurovision ordinal score lookup table for ranks 1 to 10
    # 1st -> 1.0, 2nd -> 0.9, ..., 10th -> 0.1
    RANK_POINTS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

    # Precompute per-year voters, performers, and pairwise annual rank scores
    year_voters = {}
    year_performers = {}
    year_pair_scores = {} # (year, voter, target) -> score

    for yr, yr_group in df_p.groupby("year"):
        voters = set(yr_group["from_iso3"].unique())
        performers = set(yr_group["to_iso3"].unique())
        year_voters[yr] = voters
        year_performers[yr] = performers

        for voter, vgroup in yr_group.groupby("from_iso3"):
            pos_v = vgroup[vgroup["total_points"] > 0].sort_values(by="total_points", ascending=False)
            if pos_v.empty:
                continue

            # Compute ranks (handling ties via min/average if any)
            # Group by points descending to assign standard ranks
            curr_rank = 0
            for pts, tie_group in pos_v.groupby("total_points", sort=False):
                group_len = len(tie_group)
                assigned_ranks = list(range(curr_rank, min(curr_rank + group_len, 10)))
                if assigned_ranks:
                    avg_score = sum(RANK_POINTS[r] for r in assigned_ranks) / group_len
                    for target in tie_group["to_iso3"]:
                        if target != voter:
                            year_pair_scores[(yr, voter, target)] = avg_score
                curr_rank += group_len
                if curr_rank >= 10:
                    break

    adj_matrix = pd.DataFrame(0.0, index=nodes, columns=nodes)
    years = sorted(list(year_voters.keys()))

    for u in nodes:
        for v in nodes:
            if u == v:
                continue
            co_years = sum(1 for yr in years if (u in year_voters[yr] and v in year_performers[yr]))
            if co_years > 0:
                score_sum = sum(year_pair_scores.get((yr, u, v), 0.0) for yr in years if (u in year_voters[yr] and v in year_performers[yr]))
                adj_matrix.loc[u, v] = score_sum / co_years

    return adj_matrix

def build_cepii_matrices(all_nodes):
    """
    Extract contig and comlang_off from CEPII dist_cepii.dta aligned to all_nodes index.
    Standardizes CEPII codes:
    ROM -> ROU
    YUG -> YUG (retained for P1 historical Yugoslavia)
    SCG -> Serbia and Montenegro (P2) shares contiguity/language with YUG/SRB
    SRB / MNE -> Serbia and Montenegro post-dissolution
    MCO -> Monaco borders France (FRA), official French
    """
    cepii_path = os.path.join(DATA_RAW, "dist_cepii.dta")
    df_cepii, _ = pyreadstat.read_dta(cepii_path)

    # Standardize CEPII codes
    df_cepii["iso_o"] = df_cepii["iso_o"].replace({"ROM": "ROU"})
    df_cepii["iso_d"] = df_cepii["iso_d"].replace({"ROM": "ROU"})

    contig_map = {}
    comlang_map = {}

    for _, row in df_cepii.iterrows():
        o = row["iso_o"]
        d = row["iso_d"]
        c = int(row["contig"])
        l = int(row["comlang_off"])
        contig_map[(o, d)] = c
        comlang_map[(o, d)] = l

    # Specific historical / microstate completions based on official geography and CEPII GeoDist definitions
    # Historical Yugoslavia (YUG) in P1 (pre-1992):
    # CEPII records the 1990s rump Yugoslavia (with post-secession borders like BIH, HRV, MKD).
    # Pre-1992 SFR Yugoslavia had exactly 7 land borders: ITA, AUT, HUN, ROU, BGR, GRC, ALB.
    # We clear CEPII's default post-1992 YUG borders and set its true pre-1992 neighbors.
    yug_p1_neighbors = ["ITA", "AUT", "HUN", "ROU", "BGR", "GRC", "ALB"]
    for target in all_nodes:
        if target != "YUG":
            is_neighbor = 1 if target in yug_p1_neighbors else 0
            contig_map[("YUG", target)] = is_neighbor
            contig_map[(target, "YUG")] = is_neighbor

    # Monaco (MCO): borders FRA (contig=1), official language French (comlang_off=1 with FRA, BEL, CHE, LUX)
    for target in all_nodes:
        contig_map[("MCO", target)] = 1 if target == "FRA" else contig_map.get(("MCO", target), 0)
        contig_map[(target, "MCO")] = 1 if target == "FRA" else contig_map.get((target, "MCO"), 0)
        
        mco_lang = 1 if target in ["FRA", "BEL", "CHE", "LUX"] else 0
        comlang_map[("MCO", target)] = mco_lang
        comlang_map[(target, "MCO")] = mco_lang

    # Serbia (SRB) & Serbia and Montenegro (SCG):
    # Borders: BIH, HRV, HUN, ROU, BGR, MKD, ALB, MNE
    # Language: Serbo-Croatian with BIH, HRV, MNE (comlang_off=1)
    srb_borders = ["BIH", "HRV", "HUN", "ROU", "BGR", "MKD", "ALB", "MNE"]
    srb_lang = ["BIH", "HRV", "MNE", "SCG", "YUG", "SRB"]
    for s_code in ["SRB", "SCG"]:
        for target in all_nodes:
            if target != s_code:
                if target in srb_borders:
                    contig_map[(s_code, target)] = 1
                    contig_map[(target, s_code)] = 1
                if target in srb_lang:
                    comlang_map[(s_code, target)] = 1
                    comlang_map[(target, s_code)] = 1

    # Montenegro (MNE):
    # Borders: BIH, SRB, SCG, ALB, HRV
    # Language: comlang_off=1 with SRB, SCG, BIH, HRV, YUG
    mne_borders = ["BIH", "SRB", "SCG", "ALB", "HRV"]
    mne_lang = ["BIH", "SRB", "SCG", "HRV", "YUG", "MNE"]
    for target in all_nodes:
        if target != "MNE":
            if target in mne_borders:
                contig_map[("MNE", target)] = 1
                contig_map[(target, "MNE")] = 1
            if target in mne_lang:
                comlang_map[("MNE", target)] = 1
                comlang_map[(target, "MNE")] = 1

    # Build square DataFrames indexed by all_nodes
    mat_contig = pd.DataFrame(0, index=all_nodes, columns=all_nodes)
    mat_comlang = pd.DataFrame(0, index=all_nodes, columns=all_nodes)

    for u in all_nodes:
        for v in all_nodes:
            if u != v:
                mat_contig.loc[u, v] = contig_map.get((u, v), 0)
                mat_comlang.loc[u, v] = comlang_map.get((u, v), 0)

    return mat_contig, mat_comlang

def main():
    print("Loading Eurovision Grand Final votes (1975–2025)...")
    df_votes = load_and_merge_votes()
    print(f"Loaded {len(df_votes)} Grand Final vote rows across years {df_votes['year'].min()} to {df_votes['year'].max()}.")

    # Collect all unique participating nodes across all periods
    all_nodes_set = set(df_votes["from_iso3"].unique()) | set(df_votes["to_iso3"].unique())
    all_nodes = sorted(list(all_nodes_set))
    print(f"Total unique countries across all eras: {len(all_nodes)}")

    # 1. Main pipeline (k=3) for P1, P2, P3, P4
    print("\n--- Computing Main Networks (k=3) ---")
    period_nodes = {}
    for period, (y0, y1) in ERAS.items():
        adj_topk3, edgelist_topk3, nodes, mat_topk_years, mat_co_years = compute_period_network(df_votes, period, y0, y1, k=3)
        period_nodes[period] = nodes
        print(f"{period} ({y0}–{y1}): {len(nodes)} nodes, {len(edgelist_topk3)} non-zero directed edges.")

        # Export top-k=3 weighted adjacency matrix and edgelist
        adj_topk3.to_csv(os.path.join(DATA_PROC, f"{period}_adj_topk3.csv"))
        edgelist_topk3.to_csv(os.path.join(DATA_PROC, f"{period}_edgelist_topk3.csv"), index=False)

        # Main binarized directed adjacency matrix: weight >= 0.5 AND topk_years >= 2
        # (Requires repeat voting support to filter out single-contest anomalies from late entrants)
        adj_bin_t05 = ((adj_topk3 >= 0.5) & (mat_topk_years >= 2)).astype(int)
        for diag_node in nodes:
            adj_bin_t05.loc[diag_node, diag_node] = 0
        adj_bin_t05.to_csv(os.path.join(DATA_PROC, f"{period}_adj_binary_t05.csv"))

        # Robustness binarized directed adjacency matrix: weight >= 0.33 AND topk_years >= 2
        adj_bin_t033 = ((adj_topk3 >= 0.33) & (mat_topk_years >= 2)).astype(int)
        for diag_node in nodes:
            adj_bin_t033.loc[diag_node, diag_node] = 0
        adj_bin_t033.to_csv(os.path.join(DATA_ROBUST, f"{period}_adj_binary_t033.csv"))

        # Also export unconstrained binary matrices for sensitivity reference
        adj_bin_t05_unconstrained = (adj_topk3 >= 0.5).astype(int)
        for diag_node in nodes:
            adj_bin_t05_unconstrained.loc[diag_node, diag_node] = 0
        adj_bin_t05_unconstrained.to_csv(os.path.join(DATA_ROBUST, f"{period}_adj_binary_t05_unconstrained.csv"))

        print(f"  Exported {period}_adj_topk3.csv, {period}_edgelist_topk3.csv, {period}_adj_binary_t05.csv (t=0.5, topk>=2: {adj_bin_t05.values.sum()} edges), and {period}_adj_binary_t033.csv (t=0.33, topk>=2: {adj_bin_t033.values.sum()} edges)")

    # 2. Robustness pipeline (k=2, k=4, and rank-based)
    print("\n--- Computing Robustness Networks (k=2, k=4, and rank-based) ---")
    for k_val in [2, 4]:
        for period, (y0, y1) in ERAS.items():
            adj_k, edgelist_k, _, _, _ = compute_period_network(df_votes, period, y0, y1, k=k_val)
            adj_k.to_csv(os.path.join(DATA_ROBUST, f"{period}_adj_topk{k_val}.csv"))
            edgelist_k.to_csv(os.path.join(DATA_ROBUST, f"{period}_edgelist_topk{k_val}.csv"), index=False)
            print(f"  Exported robustness/{period}_adj_topk{k_val}.csv & edgelist ({len(edgelist_k)} edges)")

    for period, (y0, y1) in ERAS.items():
        adj_rank = compute_period_rank_network(df_votes, period, y0, y1)
        adj_rank.to_csv(os.path.join(DATA_ROBUST, f"{period}_adj_rank.csv"))
        nonzero_rank = (adj_rank.values > 0).sum()
        print(f"  Exported robustness/{period}_adj_rank.csv ({nonzero_rank} non-zero entries)")

    # 3. CEPII Dyadic Attribute Matrices
    print("\n--- Building CEPII Matrices (contig & comlang_off) ---")
    # Build full matrix across all_nodes
    mat_contig_all, mat_comlang_all = build_cepii_matrices(all_nodes)
    mat_contig_all.to_csv(os.path.join(DATA_PROC, "cepii_contig_all.csv"))
    mat_comlang_all.to_csv(os.path.join(DATA_PROC, "cepii_comlang_all.csv"))

    # Also build aligned matrices per period for immediate use by Member 4 (QAP)
    for period, nodes in period_nodes.items():
        mat_contig_p = mat_contig_all.loc[nodes, nodes]
        mat_comlang_p = mat_comlang_all.loc[nodes, nodes]
        mat_contig_p.to_csv(os.path.join(DATA_PROC, f"{period}_cepii_contig.csv"))
        mat_comlang_p.to_csv(os.path.join(DATA_PROC, f"{period}_cepii_comlang.csv"))
        print(f"  Aligned CEPII matrices for {period} ({len(nodes)}x{len(nodes)})")

    print("\nNetwork construction pipeline complete!")

if __name__ == "__main__":
    main()
