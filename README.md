<p align="center">
  <img src="https://github.com/user-attachments/assets/03e693e8-d028-4899-8898-aad9d19374e7" alt="CAMS logo" width="250"/>
</p>
##  Usage

### Step-by-step

1. **Upload PGN files**:
   - Upload the PGN of the player you want to analyze (suspect).
   - Upload a PGN from a reference group (same rating, similar time control, etc.).

2. **Configure analysis parameters**:
   - `Player name`: exact name to analyze.
   - `Exclude name`: optional, to skip that name from reference group.
   - `Stockfish depth`: usually between 6 and 20.
   - `Opening plies to skip`: e.g. 6 to ignore book moves.
   - `CPU cores`: how many workers to use in parallel.

3. **Analyze results** across the 9 statistical tabs:
   - `Segmentation & KS`
   - `Histograms + QQ Plots`
   - `Fligner-Killeen`
   - `Mann–Whitney U`
   - `Control Chart`
   - `Bayesian Win-Rate + CP`
   - `Time vs. deviation`
   - `Robust Z-Score`
   - `Advantage Analysis`
##  Statistical Tests Implemented

| Method                   | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| Game Phase Segmentation  | Splits the analysis by opening, middlegame, and endgame                    |
| Kolmogorov-Smirnov Test  | Compares cumulative CP-Loss distributions (`ks_2samp()`)                   |
| Histograms + QQ Plots    | Visual distribution comparison (`go.Histogram()`, `stats.norm.ppf()`)      |
| Fligner-Killeen Test     | Tests variance homogeneity (`stats.fligner()`)                             |
| Mann-Whitney U Test      | Compares medians between groups (`stats.mannwhitneyu()`)                   |
| Control Charts           | Tracks CP-Loss trends over time (`np.nanpercentile()`)                     |
| Bayesian Win-Rate        | Models win-rate with uncertainty (`scipy.stats.beta.cdf()`)                |
| Time vs Deviation        | Analyzes clock usage irregularities (`re.search()`)                        |
| Robust Z-Score           | Outlier detection via median absolute deviation (`np.median()`, etc.)      |
| Advantage Analysis       | CP-Loss grouped by position status (winning, equal, losing)                |

##  Statistical Test Descriptions

| Test / Tab                    | Purpose                                                                                         |
|------------------------------|-------------------------------------------------------------------------------------------------|
| **Segmentation & KS Test**   | Splits CP-loss into **opening**, **middlegame**, and **endgame** and compares distributions. Performs a **Kolmogorov–Smirnov (KS)** test on the full CP-loss distribution to detect **distributional shifts** between the suspect and reference players. |
| **Histograms + QQ Plots**    | Compares the **shape** of log-transformed CP-loss distributions using **overlaid histograms** and **QQ-plots** to identify deviations from normality and mismatches in spread. |
| **Fligner–Killeen Test**     | Evaluates **variance homogeneity** between suspect and reference groups. Useful to detect inconsistent behavior or **variance inflation**, often seen in engine-assisted play. |
| **Mann–Whitney U Test**      | A non-parametric test to compare **medians** of CP-loss values. Checks if suspect performance is statistically better (lower CP-loss) than the reference group. |
| **Control Chart**            | Visualizes suspect's CP-loss **per game**, overlaid with **mean** and **percentile bounds (10th, 90th)** from the reference group. Helps detect **temporal anomalies or streaks** of high precision. |
| **Bayesian Win-Rate**        | Computes a **Bayesian posterior probability** that the suspect's win-rate is legitimate given expected win probability. Overlaid with CP-loss evolution for **dual metric tracking**. |
| **Time vs. Deviation**       | Plots **mean move time vs. standard deviation** for each game to detect **suspicious consistency** in move timings. Engine users often play with unnatural regularity. |
| **Robust Z-Score**           | Applies a **robust outlier detection** using median absolute deviation (MAD) on log(CP-loss + 1). Flags games with statistically extreme precision. |
| **Advantage-based Analysis** | Groups moves into **equal**, **winning**, or **losing** positions (±150 centipawns) and compares CP-loss across these categories. Helps reveal **inconsistent accuracy** depending on the position type. |

