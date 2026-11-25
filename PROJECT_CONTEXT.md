# Eurovision Voting Blocs Across Eras: How They Form, Persist and Dissolve (1975–2024)
- **Course:** Network Analysis (MA Digital Humanities and Digital Knowledge, UniBo) — Prof. Saverio Giallorenzo.
- **Constraints:** Max 5,000 words (~9–10 pages in LaTeX template). Grading: 20% Context/Motivation | 60% Data, Methodology, Measures & Results | 20% Interpretation & Critique. Every analysis must follow **A -> B -> C** (Phenomenon -> Network Measure -> Real-world Interpretation).
- **Professor's Feedback:** Differentiate by focusing on the longitudinal span (how blocs form, persist, and dissolve across eras). Split periods at rule changes and use binary "top-k" edges within each year to handle scoring system and participation changes.
- **Data Sources:**
  1. Main dataset (1956–2023): Spijkervet Eurovision Dataset (`votes.csv`, `contestants.csv`; ISMIR 2023 paper / Zenodo 10088892).
  2. Recent years (2024): `josago97/EurovisionDataset`.
  3. Dyadic country attributes: CEPII GeoDist (`dist_cepii.dta` -> `contig` for shared borders, `comlang_off` for shared official language).
- **Scope & Eras (Grand Final votes only, since non-qualifiers still vote in the final from 2004 onward):**
  - P1 (1975–1996): Jury only (12-10-8... system introduced in 1975; optional sub-split around 1992/1993 for Eastern European entry).
  - P2 (1997–2008): Televoting phased in (1997/1998); semi-finals introduced in 2004.
  - P3 (2009–2015): 50% jury + 50% televote combined.
  - P4 (2016–2024): Jury and televote counted separately (we use combined total for cross-era comparability; 2020 cancelled).
- **Network Construction:**
  - Directed weighted networks per period.
  - Within each year, draw a binary directed edge A -> B if B is in A's top-k (default k=3, i.e., 8, 10, 12 points).
  - Period edge weight = (years A put B in top-k) / (years A voted and B competed in the Grand Final).
  - Binarized period networks (for reciprocity & triad census): main threshold `weight >= 0.5` with `topk_years >= 2` (`{period}_adj_binary_t05.csv`) to eliminate single-year artifact ties from late joiners, and robustness threshold `weight >= 0.33` (`robustness/{period}_adj_binary_t033.csv`).
  - Robustness checks: k=2, k=4, alternative binary threshold (0.33), rank-based weights (`{period}_adj_rank.csv`), and unconstrained binary threshold.
- **Hypotheses:**
  - H1: Voting blocs exist in every period (modularity > random networks via permutations).
  - H2: Shared borders and common languages predict votes (QAP), and this effect shifts between jury and televote eras.
  - H3: Blocs are persistent (NMI, tie survival, and QAP correlation between consecutive periods).
  - H4: Entry of new countries in the 1990s–2000s created new blocs.
- **4-Person Team Split & Hand-offs (Catalina, Haopu, Maria, Nur):**
  - Member 1: Intro, Literature, Hypotheses & LaTeX Setup (~1,000 words).
  - Member 2 (Current User): Data Engineering, Network Construction (P1–P4 weighted & binarized + CEPII + robustness) & Section 2 Data/Validity/Reliability (~1,100 words).
  - Member 3: Per-Period Structural Metrics (density, reciprocity, triad census, in-degree, betweenness, core-periphery, community detection + permutations) & Section 3.1 (~1,200 words). Exports `data/03_period_structure/bloc_membership.csv` (`iso3, period, bloc_id`) for H4 so Member 4 reuses the exact same partitions.
  - Member 4: Cross-Period Dynamics (NMI using Member 3's `bloc_membership.csv`, tie survival), QAP Hypothesis Testing & Section 3.2 + Section 4 Critique/Limitations (~1,300 words).
- **Milestones:**
  - Milestone 1 (Nov 20): Member 2 pushes all clean datasets (`data/02_network_construction/`) to GitHub.
  - Milestone 2 (Nov 25): Member 3 shares per-period metrics, plots, and `bloc_membership.csv`; Member 4 shares QAP & persistence results.
  - Milestone 3 (Nov 29): Complete first draft in Overleaf following A -> B -> C.
  - Final Submission (Dec 2): Joint editing, word-count check (< 5,000 words), and LaTeX polish.
- **Course Lecture Concepts for Section 2 (Data, Validity & Reliability):**
  - Boundary specification: Nominalist ("etic", objective third-person rule: EBU Grand Final participants/voters 1975–2024) vs. Realist ("emic", self-recognition).
  - Validity ("are we measuring what we intend to measure?"): Omission errors (missing edges/nodes distorting centrality/connectivity—avoided in post-2004 finals since non-qualifiers still vote) & Commission errors (erroneous inclusion of nodes/edges).
  - Reliability ("same conditions, same results"): Threats from secondary sources/data mining (archival digitization biases in early contests) and formatting errors (inconsistent country codes across datasets).
