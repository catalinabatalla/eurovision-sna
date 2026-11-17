"""
02b_data_summary.py
Compute descriptive summary statistics across all eras (P1-P4)
and generate both console summary and LaTeX table in paper/table_dataset_summary.tex.
"""
import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(BASE_DIR, "data", "raw")
DATA_PROC = os.path.join(BASE_DIR, "data", "processed")
DATA_ROBUST = os.path.join(DATA_PROC, "robustness")
PAPER_DIR = os.path.join(BASE_DIR, "paper")

os.makedirs(PAPER_DIR, exist_ok=True)

ERAS = {
    "P1": {"span": "1975--1996", "desc": "Jury only", "y0": 1975, "y1": 1996},
    "P2": {"span": "1997--2008", "desc": "Televote rollout", "y0": 1997, "y1": 2008},
    "P3": {"span": "2009--2015", "desc": "50/50 Jury and Televote", "y0": 2009, "y1": 2015},
    "P4": {"span": "2016--2024", "desc": "Separate voting (combined)", "y0": 2016, "y1": 2024},
}

def count_editions(y0, y1):
    """Count number of contest finals in [y0, y1] from raw votes."""
    df_spijk = pd.read_csv(os.path.join(DATA_RAW, "votes_spijkervet.csv"))
    df_josago = pd.read_csv(os.path.join(DATA_RAW, "votes_josago.csv"))

    years_spijk = set(df_spijk[(df_spijk["year"] >= y0) & (df_spijk["year"] <= y1) & (df_spijk["round"] == "final")]["year"])
    years_josago = set(df_josago[(df_josago["year"] >= y0) & (df_josago["year"] <= y1) & (df_josago["round"] == "final")]["year"])
    return len(years_spijk | years_josago)

def main():
    rows = []

    print(f"{'Period':<6} | {'Span':<11} | {'Finals':<6} | {'Nodes':<5} | {'Edges (k=3)':<11} | {'Mean W':<8} | {'Bin t>=0.5':<10} | {'Bin t>=0.33':<11} | {'Contig %':<8} | {'ComLang %':<9}")
    print("-" * 105)

    for period, meta in ERAS.items():
        n_finals = count_editions(meta["y0"], meta["y1"])

        # 1. Weighted adj (top-k=3)
        adj_topk3 = pd.read_csv(os.path.join(DATA_PROC, f"{period}_adj_topk3.csv"), index_col=0)
        n_nodes = len(adj_topk3)

        # Non-zero directed edges and mean weight
        w_values = adj_topk3.values[~np.eye(n_nodes, dtype=bool)]
        nonzero_w = w_values[w_values > 0]
        n_nonzero_edges = len(nonzero_w)
        mean_weight = float(np.mean(nonzero_w)) if len(nonzero_w) > 0 else 0.0

        # 2. Binary t >= 0.5
        adj_bin_05 = pd.read_csv(os.path.join(DATA_PROC, f"{period}_adj_binary_t05.csv"), index_col=0)
        n_bin_05 = int(adj_bin_05.values.sum())

        # 3. Binary t >= 0.33 (robustness)
        adj_bin_033 = pd.read_csv(os.path.join(DATA_ROBUST, f"{period}_adj_binary_t033.csv"), index_col=0)
        n_bin_033 = int(adj_bin_033.values.sum())

        # 4. CEPII contig & comlang
        mat_contig = pd.read_csv(os.path.join(DATA_PROC, f"{period}_cepii_contig.csv"), index_col=0)
        mat_comlang = pd.read_csv(os.path.join(DATA_PROC, f"{period}_cepii_comlang.csv"), index_col=0)

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

    df_summary = pd.DataFrame(rows)

    # Generate LaTeX table
    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Eurovision Voting Networks Dataset Summary across Eras (1975--2024)}",
        r"\label{tab:dataset_summary}",
        r"\small",
        r"\begin{tabular}{l l c c c c c c c c}",
        r"\hline\hline",
        r"Period & Voting System & Years & Finals & $N$ & $E_{\text{weighted}}$ & Mean $W$ & $E_{t \ge 0.5}$ & $E_{t \ge 0.33}$ & Contig. (\%) & ComLang (\%) \\",
        r"\hline",
    ]

    for r in rows:
        line = (
            f"{r['Period']} & {r['Description']} & {r['Span']} & {r['Finals']} & {r['Nodes']} & "
            f"{r['WeightedEdges']} & {r['MeanWeight']:.3f} & {r['BinT05']} & {r['BinT033']} & "
            f"{r['ContigPct']:.1f}\\% & {r['ComLangPct']:.1f}\\% \\\\"
        )
        latex_lines.append(line)

    latex_lines.extend([
        r"\hline\hline",
        r"\end{tabular}",
        r"\begin{tablenotes}",
        r"\footnotesize",
        r"\item \textit{Note:} $N$ represents active Grand Final participants/voters in the era; $E_{\text{weighted}}$ denotes directed ties where $W(A \to B) > 0$ for top-$k=3$; $E_{t \ge 0.5}$ and $E_{t \ge 0.33}$ are binarized edges at the main ($0.50$) and robustness ($0.33$) persistence thresholds requiring at least two top-$k$ voting occurrences ($\text{top-}k\text{ years} \ge 2$) to eliminate single-year artifact ties from late joiners; Contig. and ComLang show the percentage of directed dyads sharing a land border or official language from CEPII GeoDist.",
        r"\end{tablenotes}",
        r"\end{table}",
    ])

    latex_content = "\n".join(latex_lines) + "\n"
    latex_path = os.path.join(PAPER_DIR, "table_dataset_summary.tex")
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex_content)

    print(f"\nLaTeX table successfully exported to: {latex_path}")

if __name__ == "__main__":
    main()
