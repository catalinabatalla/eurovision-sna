"""Helper functions shared by the project notebooks."""

import itertools
import os

import networkx as nx
import numpy as np
import pandas as pd
import pyreadstat
import requests
import matplotlib.patheffects as pe
from networkx.algorithms import community as nxc

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
            # In case of draws, include candidates with points > 0 matching the k-th rank threshold
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


def binarize_network(adj: pd.DataFrame, topk_years: pd.DataFrame, tau: float, min_years: int = 1) -> pd.DataFrame:
    """Binary network of stable ties: A -> B is 1 when w_AB >= tau and B was in A's top k in at least min_years years."""
    binary = ((adj >= tau) & (topk_years >= min_years)).astype(int)
    return binary.where(~np.eye(len(binary), dtype=bool), 0)


def compute_period_rank_network(df_votes, period_name, year_min, year_max):
    """
    Rank-based version of compute_period_network: instead of "in the top k or not",
    every position in a vote gets a score (1st place -> 1.0, 2nd -> 0.9, ..., 10th -> 0.1, else 0).
    The weight of A -> B is the average score A gave B over the years in which A voted and B was a finalist.
    Countries in a draw (same points) share the average score of their positions.
    Returns the weighted adjacency matrix (DataFrame).
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

            # Compute ranks (handling draws via min/average if any)
            # Group by points descending to assign standard ranks
            curr_rank = 0
            for pts, draw_group in pos_v.groupby("total_points", sort=False):
                group_len = len(draw_group)
                assigned_ranks = list(range(curr_rank, min(curr_rank + group_len, 10)))
                if assigned_ranks:
                    avg_score = sum(RANK_POINTS[r] for r in assigned_ranks) / group_len
                    for target in draw_group["to_iso3"]:
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


def sensitivity_grid(networks: dict, taus=(0.33, 0.5), min_years_values=(1, 2, 3, 4)) -> pd.DataFrame:
    """
    Binarise every network with every combination of tau and minimum years, and measure the result.
    networks: {(period, k): (weighted adjacency matrix, topk_years matrix)}.
    Returns one row per (period, k, tau, min_years) with the number of stable ties, isolated countries,
    reciprocity and reciprocity divided by density.
    """
    rows = []
    for (period, k_val), (adj, topk_years) in sorted(networks.items()):
        for tau in taus:
            for min_years in min_years_values:
                G = nx.from_pandas_adjacency(binarize_network(adj, topk_years, tau, min_years), create_using=nx.DiGraph)
                dens = nx.density(G)
                rec = nx.reciprocity(G) if G.number_of_edges() else np.nan
                rows.append({"period": period, "k": k_val, "tau": tau, "min_years": min_years,
                             "stable_ties": G.number_of_edges(),
                             "isolated": len(list(nx.isolates(G))),
                             "reciprocity": rec,
                             "reciprocity_vs_random": rec / dens if dens else np.nan})
    return pd.DataFrame(rows)


def channel_votes(df_spijk: pd.DataFrame, df_josago: pd.DataFrame, iso2_to_iso3: dict) -> pd.DataFrame:
    """
    Grand Final votes of P4 with the jury and televote points kept separately:
    2016-2023 from Spijkervet, 2024 from EurovisionDataset.
    Non-countries (Rest of the World) are removed and country codes are converted to ISO3.
    """
    cols = ["year", "from_country_id", "to_country_id", "jury_points", "tele_points", "total_points"]
    df = pd.concat([df_spijk[(df_spijk["round"] == "final") & df_spijk["year"].between(2016, 2023)][cols],
                    df_josago[(df_josago["round"] == "final") & (df_josago["year"] == 2024)][cols]], ignore_index=True)
    for c in ["from_country_id", "to_country_id"]:
        df[c] = df[c].astype(str).str.strip().str.upper()
    df = df[~df["from_country_id"].isin(["WLD", "ROW"]) & ~df["to_country_id"].isin(["WLD", "ROW"])]
    df["from_iso3"] = df["from_country_id"].map(iso2_to_iso3)
    df["to_iso3"] = df["to_country_id"].map(iso2_to_iso3)
    for c in ["jury_points", "tele_points", "total_points"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df


def build_channel_networks(df_ch: pd.DataFrame, year_min: int, year_max: int) -> dict:
    """
    One network per voting channel (jury, televote), built like the main networks (k = 3)
    but with jury_points or tele_points instead of total_points.
    Returns {channel: (weighted adjacency matrix, binary network with tau = 0.5 and 2 years, nodes, co_years matrix)}.
    """
    results = {}
    for channel, col in {"jury": "jury_points", "televote": "tele_points"}.items():
        adj, _, nodes, topk_years, co_years = compute_period_network(df_ch.assign(total_points=df_ch[col]), "P4", year_min, year_max, k=3)
        results[channel] = (adj, binarize_network(adj, topk_years, tau=0.5, min_years=2), nodes, co_years)
    return results


def build_cepii_matrices(all_nodes, cepii_path):
    """
    Extract contig and comlang_off from CEPII dist_cepii.dta aligned to all_nodes index.
    Standardizes CEPII codes:
    ROM -> ROU
    YUG -> YUG (retained for P1 historical Yugoslavia)
    SCG -> Serbia and Montenegro (P2) shares contiguity/language with YUG/SRB
    SRB / MNE -> Serbia and Montenegro post-dissolution
    MCO -> Monaco borders France (FRA), official French
    """
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


def count_editions(y0, y1, data_raw):
    """Count number of contest finals in [y0, y1] from raw votes."""
    df_spijk = pd.read_csv(os.path.join(data_raw, "votes_spijkervet.csv"))
    df_josago = pd.read_csv(os.path.join(data_raw, "votes_josago.csv"))

    years_spijk = set(df_spijk[(df_spijk["year"] >= y0) & (df_spijk["year"] <= y1) & (df_spijk["round"] == "final")]["year"])
    years_josago = set(df_josago[(df_josago["year"] >= y0) & (df_josago["year"] <= y1) & (df_josago["round"] == "final")]["year"])
    return len(years_spijk | years_josago)


def summary_statistics(summary_eras, data_raw, data_proc, data_robust):
    """
    Descriptive statistics of the exported networks of each period (finals, countries, weighted edges,
    mean weight, stable ties at tau = 0.5 and 0.33, % of pairs sharing a border / an official language).
    Prints them as a text table and returns one dict per period.
    """
    rows = []

    print(f"{'Period':<6} | {'Span':<11} | {'Finals':<6} | {'Nodes':<5} | {'Edges (k=3)':<11} | {'Mean W':<8} | {'Bin t>=0.5':<10} | {'Bin t>=0.33':<11} | {'Contig %':<8} | {'ComLang %':<9}")
    print("-" * 105)

    for period, meta in summary_eras.items():
        n_finals = count_editions(meta["y0"], meta["y1"], data_raw)

        # 1. Weighted adj (top-k=3)
        adj_topk3 = pd.read_csv(os.path.join(data_proc, f"{period}_adj_topk3.csv"), index_col=0)
        n_nodes = len(adj_topk3)

        # Non-zero directed edges and mean weight
        w_values = adj_topk3.values[~np.eye(n_nodes, dtype=bool)]
        nonzero_w = w_values[w_values > 0]
        n_nonzero_edges = len(nonzero_w)
        mean_weight = float(np.mean(nonzero_w)) if len(nonzero_w) > 0 else 0.0

        # 2. Binary t >= 0.5
        adj_bin_05 = pd.read_csv(os.path.join(data_proc, f"{period}_adj_binary_t05.csv"), index_col=0)
        n_bin_05 = int(adj_bin_05.values.sum())

        # 3. Binary t >= 0.33 (robustness)
        adj_bin_033 = pd.read_csv(os.path.join(data_robust, f"{period}_adj_binary_t033.csv"), index_col=0)
        n_bin_033 = int(adj_bin_033.values.sum())

        # 4. CEPII contig & comlang
        mat_contig = pd.read_csv(os.path.join(data_proc, f"{period}_cepii_contig.csv"), index_col=0)
        mat_comlang = pd.read_csv(os.path.join(data_proc, f"{period}_cepii_comlang.csv"), index_col=0)

        total_pairs = n_nodes * (n_nodes - 1)
        contig_pairs = int(mat_contig.values[~np.eye(n_nodes, dtype=bool)].sum())
        comlang_pairs = int(mat_comlang.values[~np.eye(n_nodes, dtype=bool)].sum())

        pct_contig = (contig_pairs / total_pairs * 100) if total_pairs > 0 else 0.0
        pct_comlang = (comlang_pairs / total_pairs * 100) if total_pairs > 0 else 0.0

        rows.append({
            "Period": period,
            "Description": meta["desc"],
            "Span": meta["span"],
            "Finals": n_finals,
            "Nodes": n_nodes,
            "WeightedEdges": n_nonzero_edges,
            "MeanWeight": mean_weight,
            "BinT05": n_bin_05,
            "BinT033": n_bin_033,
            "ContigPct": pct_contig,
            "ComLangPct": pct_comlang,
        })

        print(f"{period:<6} | {meta['span']:<11} | {n_finals:<6} | {n_nodes:<5} | {n_nonzero_edges:<11} | {mean_weight:<8.3f} | {n_bin_05:<10} | {n_bin_033:<11} | {pct_contig:<7.2f}% | {pct_comlang:<8.2f}%")

    return rows


def best_louvain(G, n_runs=100):
    """Louvain with n_runs different seeds; returns the partition (list of sets) with the highest modularity and its Q."""
    best_blocs, best_q = None, -1
    for s in range(n_runs):
        blocs = nxc.louvain_communities(G, weight="weight", seed=s)
        q = nxc.modularity(G, blocs, weight="weight")
        if q > best_q:
            best_q, best_blocs = q, blocs
    return best_blocs, best_q


def shuffle_rows(A, rng):
    """Null network: each country keeps the weights it gives, but they are shuffled among the other countries."""
    M = A.values.copy()
    n = M.shape[0]
    R = np.zeros_like(M)
    for i in range(n):
        others = [j for j in range(n) if j != i]
        R[i, others] = rng.permutation(M[i, others])
    return pd.DataFrame(R, index=A.index, columns=A.columns)


def uman_random_graph(nodes, n_mutual, n_asym, rng):
    """U|MAN random graph: the same numbers of mutual, asymmetric and null pairs as the real network, on random pairs."""
    pairs = list(itertools.combinations(nodes, 2))
    order = rng.permutation(len(pairs))
    R = nx.DiGraph()
    R.add_nodes_from(nodes)
    for k in order[:n_mutual]:
        u, v = pairs[k]
        R.add_edge(u, v)
        R.add_edge(v, u)
    for k in order[n_mutual:n_mutual + n_asym]:
        u, v = pairs[k]
        if rng.random() < 0.5:
            R.add_edge(u, v)
        else:
            R.add_edge(v, u)
    return R


def coreness_scores(W):
    """Coreness of each country (0-1): leading eigenvector of the symmetrised weights. Returns (coreness, symmetrised matrix)."""
    S = (W + W.T) / 2
    vals, vecs = np.linalg.eigh(S)
    c = np.abs(vecs[:, -1])
    return c / c.max(), S


def cp_fit(S, c):
    """Core-periphery fit: correlation between the real ties and the ideal pattern c_i * c_j."""
    mask = ~np.eye(len(c), dtype=bool)
    ideal = np.outer(c, c)
    return np.corrcoef(S[mask], ideal[mask])[0, 1]


def off(M):
    """All the values of a square matrix except the diagonal (a country with itself)."""
    return M[~np.eye(len(M), dtype=bool)]       # all pairs except a country with itself


def qap_r(Y, X, n_perm=2000, seed=2024):
    """QAP test: correlation between two country-by-country matrices and its p-value, shuffling the countries of Y."""
    rng = np.random.default_rng(seed)
    r = np.corrcoef(off(Y), off(X))[0, 1]
    null = []
    for _ in range(n_perm):
        idx = rng.permutation(len(Y))
        null.append(np.corrcoef(off(Y[np.ix_(idx, idx)]), off(X))[0, 1])
    return r, (np.sum(np.array(null) >= r) + 1) / (n_perm + 1)


# figure style of notebook 04 (same as notebook 03)
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#b9b8b2"
GRID = "#ecebe7"
CH_COLOR = {"jury": "#5b45b5", "televote": "#c98500", "combined": "#8a8984"}
CH_LABEL = {"jury": "Jury", "televote": "Televote", "combined": "Combined"}
CH_TICK = {"jury": "Jury", "televote": "Tele-\nvote", "combined": "Com-\nbined"}

def style(ax, title, subtitle, ylabel=None):
    ax.set_title(title, loc="left", fontsize=9, fontweight="bold", color=INK, pad=14)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=6.8, color=INK2, va="bottom")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=7.4, color=INK2)
    ax.tick_params(axis="y", labelsize=6.5, colors=INK2, length=0)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)

def label(ax, x, y, text, dy=4.5, color=INK):
    ax.annotate(text, (x, y), xytext=(0, dy), textcoords="offset points", ha="center", va="bottom",
                fontsize=7, color=color, fontweight="bold",
                path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])
