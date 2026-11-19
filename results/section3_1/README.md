# Section 3.1: per-period structural analysis of Eurovision voting blocs

Code, results and figures for **Section 3.1 (Structural Evolution & Voting Blocs Across Eras)** of the group project *How Eurovision voting blocs form, persist and dissolve across eras (1975–2024)*, Network Analysis course, University of Bologna (DHDK), 2026/27.

Author of this part: **HAOPU YIN** (Member 3).
Data preparation and network construction: Catalina Batalla (see `src/01_download_data.py` and `src/02_build_networks.py`).

## Where the files are

This folder holds the result tables of Section 3.1. The other files of this part are stored with the rest of the project:

```
eurovision-sna/
├── data/processed/bloc_membership.csv   # bloc of each country in each period (used in Section 3.2)
├── src/03_period_structure.ipynb        # all analyses and figures of Section 3.1 (Google Colab)
├── results/section3_1/                  # result tables (this folder)
└── paper/figures/                       # Figures 1–3 of the report (PDF and PNG)
```

## Data

The notebook reads the processed networks directly from this repository:

- Folder: `data/processed/`
- Version used: commit `ef6dca8` (17 November 2026)
- Main network per period: `{period}_adj_topk3.csv` (directed, weighted; weight = share of years in which country A placed B in its top three, out of the years both were in the final)
- Binary network of stable ties: `{period}_adj_binary_t05.csv` (weight ≥ 0.5 and at least two top-three years)
- Robustness networks: `robustness/` (k = 2, k = 4, rank-based weights, binary threshold 0.33)

Periods: P1 1975–1996 (jury only), P2 1997–2008 (televoting introduced), P3 2009–2015 (50/50 jury and televote), P4 2016–2024 (separate jury and televote).

## Analyses (in notebook order)

1. Louvain community detection on the directed weighted networks (best of 100 fixed seeds)
2. Modularity permutation test against 500 null networks per period (outgoing weights reshuffled; H1)
3. Bloc matching between consecutive periods with Jaccard overlap and event classification: birth, continuation, merge, death (H4)
4. Density and reciprocity of the binary networks
5. Triad census against U|MAN random graphs
6. Normalised in-strength, stable supporters and betweenness centrality
7. Continuous core–periphery fit (leading eigenvector of the symmetrised network) against the same null networks
8. Robustness checks: k = 2, k = 4, rank-based weights, P1 split at 1993, binary threshold 0.33, Jaccard thresholds 0.2 and 0.5
9. Figures 1–3
10. Sensitivity check of the permutation test (observed Q re-estimated with the same number of Louvain runs as the null networks)

## Result files (this folder)

| File | Content |
|---|---|
| `../../data/processed/bloc_membership.csv` | Bloc of each country in each period (`iso3`, `period`, `bloc_id`) |
| `modularity_permutation_test.csv` | Observed modularity, null mean and SD, z-score, p-value |
| `bloc_overlap.csv` | Jaccard overlap between blocs of consecutive periods |
| `bloc_events.csv` | Event classification of each bloc and number of new entrants |
| `basic_structure.csv` | Density, reciprocity, mutual pairs and isolates per period |
| `triad_census.csv`, `triad_groups.csv`, `triad_groups_UMAN.csv` | Triad census and comparison with random graphs |
| `centrality.csv` | Normalised in-strength, stable supporters and betweenness per country and period |
| `core_periphery_fit.csv`, `coreness.csv` | Core–periphery fit per period and coreness per country |
| `robustness_weighted.csv` | Modularity, NMI and core–periphery results for alternative specifications |
| `robustness_binary_t033.csv` | Reciprocity and triad results at binary threshold 0.33 |
| `robustness_jaccard.csv` | Bloc matching at Jaccard thresholds 0.2, 0.3 and 0.5 |

Bloc IDs are labels within each period only; bloc 1 in P1 is not the same bloc as bloc 1 in P2.

## Figures (`paper/figures/`)

- `fig1_bloc_maps`: voting blocs in the four periods on a schematic map
- `fig2_bloc_evolution`: evolution of blocs across periods (Jaccard matching and events)
- `fig3_structure_over_time`: modularity, reciprocity and core–periphery fit by period, with the range across robustness specifications

## Reproducibility

- All stochastic steps use fixed seeds: 100 seeds (0–99) for Louvain on the observed networks, and seed 2024 for every null model and permutation test.
- Python 3 on Google Colab, NetworkX 3.7, pandas, NumPy, Matplotlib.
- The P1 split uses `compute_period_network` from `src/02_build_networks.py` (the notebook clones the repository and installs `pyreadstat`).
- To rerun: open `src/03_period_structure.ipynb` in Google Colab and select *Run all* (about 8–10 minutes).

## References for the methods

- Greene, D., Doyle, D., & Cunningham, P. (2010). Tracking the evolution of communities in dynamic social networks. In *Proceedings of the 2010 International Conference on Advances in Social Networks Analysis and Mining (ASONAM 2010)* (pp. 176–183). IEEE.
- Palla, G., Barabási, A.-L., & Vicsek, T. (2007). Quantifying social group evolution. *Nature*, 446(7136), 664–667. https://doi.org/10.1038/nature05670
