# How to Plan an Engaging Cocktail Party: The Edge Fair Densest Subgraph Problem

Research code for experiment reproduction on connectivity-fair densest-subgraph extraction.

The paper **“How to Plan an Engaging Cocktail Party: The Edge Fair Densest Subgraph Problem”** studies fairness in densest-subgraph discovery from a **connectivity** perspective rather than a **balanced node-representation** perspective. It introduces **group density** and **group-diverse density** as group-aware connectivity measures, and defines two max-min objectives: **edge-fair density** (`eFairD`) and **edge-fair diverse density** (`eFairDivD`). The paper develops greedy, spectral, and deep methods for these objectives and evaluates them on synthetic and real attributed graphs, showing improved worst-group connectivity while preserving competitive overall density. 

The repository includes implementations of:
- density-only baselines,
- diverse-density extraction,
- fairness-aware greedy methods,
- spectral methods, and
- deep GCN-based methods.


---

## What is implemented?

Given an attributed graph \(G=(V,E)\), where nodes are partitioned into groups, the code supports the following objectives:

- **Density**: `D(S) = |E(S)| / |S|`
- **Group density** for group \(g_i\): `D_i(S) = |E_i(S)| / |S|`
- **Diverse density**: `D_div(S) = |E_div(S)| / |S|`
- **Group-diverse density** for group \(g_i\): `D_div,i(S) = |E_div,i(S)| / |S|`
- **Edge-fair density**: `eFairD(S) = min_i D_i(S)`
- **Edge-fair diverse density**: `eFairDivD(S) = min_i D_div,i(S)`

At a high level, the repository supports both:
- **single-objective extraction**, where the goal is to maximize diverse density, and
- **fairness-aware extraction**, where the goal is to improve the weakest group-specific connectivity score.

This matches the paper’s view of connectivity fairness through two complementary notions:
- preserving sufficient **group-incident connectivity** for every group, and
- preserving sufficient **cross-group connectivity** for every group. 

---

## Repository layout

```text
.
├── main.py
├── greedy_algortihm.py
├── spectral_densest_subgraph.py
├── Deep Dense subgraph/
│   ├── deepDenseSubgraph.py
│   └── tools/
│       ├── deep_densest_models.py
│       ├── gcn.py
│       ├── metrics.py
│       └── utils.py
└── datasets/
    └── <dataset_folder>/
```

### Main files

- **`main.py`**
  Main experiment runner. It loads datasets, runs all configured methods, computes summary metrics, and writes results to CSV.

- **`greedy_algortihm.py`**
  Greedy peeling implementations for:
  - density,
  - diverse density,
  - edge-fair density (`eFairD`), and
  - edge-fair diverse density (`eFairDivD`).

- **`spectral_densest_subgraph.py`**
  Eigenvector-sweep implementation for spectral densest-subgraph extraction.

- **`Deep Dense subgraph/deepDenseSubgraph.py`**
  Deep learning implementations based on a GCN encoder and differentiable Rayleigh-style objectives.

- **`Deep Dense subgraph/tools/deep_densest_models.py`**
  TensorFlow layers for the deep densest-subgraph objectives.

---

## Implemented methods

The main script currently runs the following methods:

| Code name | Paper-style name | Type | Target objective |
|---|---|---|---|
| `greedy` | Greedy | Greedy | Density |
| `spectral` | Spectral | Spectral | Density |
| `deep` | Deep | Deep  | Density |
| `greedy_diverse` | GreedyDiverse | Greedy | Diverse Density |
| `spectral_diverse` | SpectralDiverse | Spectral | Diverse Density |
| `deepDiverse` | DeepDiverse | Deep  | Diverse Density |
| `maxmin_greedy` | GreedyGroupFairness | Greedy | `eFairD` |
| `deepGroupMaxMin` | DeepGroupFairness | Deep  | `eFairD` |
| `maxmin_diverse_greedy` | GreedyGroupDiverseFairness | Greedy | `eFairDivD` |
| `deepGroupDiverseMaxMin` | DeepGroupDiverseFairness | Deep  | `eFairDivD` |

---

## Requirements

A typical Python environment for this repository includes:

- Python 3.9+
- `numpy`
- `pandas`
- `scipy`
- `networkx`
- `scikit-learn`
- `matplotlib`
- `tensorflow`

Example setup:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
# source venv/bin/activate

pip install --upgrade pip
pip install numpy pandas scipy networkx scikit-learn matplotlib tensorflow
```

---

## Input format

The main runner expects a directory under `datasets/` for each graph.

A dataset folder should contain at least:

```text
<dataset_folder>/
├── <dataset_name>.edgelist
├── <dataset_name>_attributes.csv
└── planted_subgraphs.csv          # optional but recommended for synthetic data
```

### 1) Edge list

Space-separated edge list:

```text
u v
u w
v w
```

### 2) Attribute file

CSV file with a node column and an `attribute` column.

Recommended format:

```csv
node,attribute
0,0
1,2
2,1
```

The loaders are flexible with node-column naming and accept variants such as:
- `node`
- `nodes`
- `node_id`
- `node_idx`

### 3) Ground-truth planted sets (optional)

For synthetic datasets, `planted_subgraphs.csv` can be used to evaluate overlap against known planted sets.

Example format:

```csv
set_name,node
S1,1
S1,3
S2,14
S2,27
```

---

## Running the code

### Quickstart

From the repository root:

```bash
python main.py
```

The script scans all folders under `datasets/`, runs the configured methods, and writes summary results.

### Main configuration

A few useful constants are defined near the top of `main.py`:

- **`root_name`**: dataset root directory (default: `datasets`)
- **`STest`**: planted target set name used for synthetic evaluation (default: `S1`)
- **`N_RUNS_PER_METHOD`**: number of repetitions per method
- **`OUT_ROOT`**: output directory for extracted subgraphs

If you want to run a different dataset root or evaluate against a different planted set, edit these constants in `main.py`.

---

## Outputs

The code writes two types of outputs.

### 1) Per-run extracted subgraphs

For each method and dataset, the selected node set is written to:

```text
synthetic_results/<root_name>/<method>/<dataset_folder>/densest_subgraph_runXX.csv
```

These files contain the extracted node ids and, when available, the node attributes.

### 2) Summary results

After all runs, `main.py` writes an aggregated CSV of method-level metrics:

```text
<root_name>_results_<STest>.csv
```

Typical columns include:
- `density`
- `group_density_min`, `group_density_max`, `group_density_avg`
- `group_diverse_density_min`, `group_diverse_density_max`, `group_diverse_density_avg`
- `diverse_density`
- `subgraph_size`
- `runtime_sec`
- overlap metrics with planted ground truth, such as `gt_jaccard`, `gt_precision`, `gt_recall`, and `gt_f1`

---

## Deep models

The deep variants use a GCN encoder followed by a learnable scoring layer and optimize differentiable Rayleigh-style relaxations of the target objective.

The current defaults in the repository are:
- optimizer: **Adam**
- learning rate: **0.01**
- training epochs: **1000**

Implemented deep variants include:
- **Deep** for density,
- **DeepDiverse** for diverse density,
- **DeepGroupFairness** for `eFairD`, and
- **DeepGroupDiverseFairness** for `eFairDivD`.

---

## Running time and variability results

The tables below reproduce the running-time and variability results reported in the paper. Values are shown as **mean ± standard deviation** over 10 runs. Running times are in seconds, and `|S|` denotes the number of nodes in the selected subgraph. A dash (—) indicates that no result is reported for that method and dataset.

The experiments were run on a system with 128 CPU threads, 1 TiB RAM, and two NVIDIA H200 GPUs.

### Running times

| Method | Deezer | Facebook_c | Facebook_t | Twitch | DBLP | Pokec-a | Pokec-g |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Greedy | 0.113 ± 0.012 | 0.056 ± 0.001 | 0.048 ± 0.001 | 0.628 ± 0.021 | 0.136 ± 0.002 | 63.08 ± 0.36 | 64.88 ± 4.07 |
| GreedyGroupFairness | 1.071 ± 0.004 | 0.571 ± 0.041 | 0.617 ± 0.024 | 8.042 ± 0.115 | 2.090 ± 0.046 | 456.93 ± 4.19 | 476.66 ± 4.80 |
| GreedyDiverse | 0.460 ± 0.046 | 0.076 ± 0.012 | 0.209 ± 0.018 | 0.313 ± 0.043 | 0.410 ± 0.028 | 94.71 ± 2.80 | 220.28 ± 4.29 |
| GreedyGroupDiverseFairness | 1.104 ± 0.056 | 0.557 ± 0.034 | 0.620 ± 0.025 | 12.20 ± 0.34 | 2.110 ± 0.038 | 470.56 ± 6.93 | 459.66 ± 4.63 |
| Spectral | 0.201 ± 0.002 | 0.0622 ± 0.0004 | 0.056 ± 0.012 | 0.610 ± 0.042 | 0.212 ± 0.002 | 123.48 ± 6.97 | 126.00 ± 3.10 |
| Deep | 17.16 ± 7.28 | 24.51 ± 8.26 | 27.49 ± 6.34 | 22.58 ± 9.38 | 27.93 ± 6.40 | 385.51 ± 9.25 | 384.43 ± 12.86 |
| DeepGroupFairness | 6.93 ± 0.44 | 6.39 ± 0.15 | 3.15 ± 0.21 | 8.90 ± 0.40 | 3.68 ± 0.25 | 188.99 ± 6.56 | 186.83 ± 4.50 |
| DeepDiverse | 28.02 ± 5.72 | 17.00 ± 7.39 | 25.52 ± 7.66 | 20.40 ± 8.76 | 27.89 ± 6.06 | 114.53 ± 5.16 | 223.23 ± 7.75 |
| DeepGroupDiverseFairness | 2.87 ± 0.15 | 6.51 ± 0.20 | 6.81 ± 0.21 | 7.60 ± 1.72 | 6.17 ± 1.80 | 148.63 ± 2.51 | 179.55 ± 3.30 |
| Adam | 0.787 ± 0.272 | 0.584 ± 0.038 | 0.524 ± 0.051 | 2.193 ± 0.078 | 0.715 ± 0.018 | 80.84 ± 9.62 | 85.35 ± 4.11 |
| AdamGroupFairness | 1.279 ± 0.294 | 0.816 ± 0.086 | 1.261 ± 0.173 | 3.757 ± 0.092 | 1.790 ± 0.020 | 124.90 ± 5.73 | 111.42 ± 20.13 |
| AdamDiverse | 0.680 ± 0.034 | 0.509 ± 0.049 | 0.528 ± 0.022 | 0.700 ± 0.038 | 0.651 ± 0.023 | 16.96 ± 0.17 | 19.88 ± 0.12 |
| AdamGroupDiverseFairness | 0.989 ± 0.081 | 0.823 ± 0.075 | 1.226 ± 0.113 | 3.464 ± 0.044 | 1.705 ± 0.036 | 96.82 ± 1.52 | 110.41 ± 3.29 |
| Balance Spectral | 0.395 ± 0.083 | 0.045 ± 0.001 | -- | -- | -- | 23.97 ± 0.60 | 21.85 ± 0.40 |
| DDSP | 13.37 ± 0.21 | 0.347 ± 0.035 | 0.497 ± 0.019 | 39.62 ± 0.47 | 10.58 ± 0.22 | -- | -- |
| ColApproxGroup | 0.574 ± 0.020 | 0.265 ± 0.006 | 0.286 ± 0.008 | 5.21 ± 0.12 | 1.14 ± 0.05 | 4081.62 ± 500.83 | 3290.56 ± 40.88 |
| ColApproxDiv | 0.228 ± 0.001 | 0.065 ± 0.015 | 0.193 ± 0.018 | 0.327 ± 0.069 | 0.247 ± 0.029 | 55.86 ± 9.42 | 120.60 ± 12.87 |

### Variability of fairness-aware Deep and Adam methods

The following table reports the score, subgraph size, and running time for the two fairness objectives, `eFairD` and `eFairDivD`.

| Dataset | Method | eFairD score | eFairD \|S\| | eFairD time (s) | eFairDivD score | eFairDivD \|S\| | eFairDivD time (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Deezer | Deep | 5.123 ± 0.008 | 1189.7 ± 26.4 | 6.9 ± 0.4 | 3.905 ± 0.007 | 1041.2 ± 31.6 | 2.9 ± 0.1 |
| Deezer | Adam | 5.785 ± 0.000 | 242.0 ± 0.0 | 1.28 ± 0.29 | 4.874 ± 0.000 | 183.0 ± 0.0 | 0.99 ± 0.08 |
| Facebook_c | Deep | 12.130 ± 0.142 | 152.0 ± 7.5 | 6.4 ± 0.2 | 11.748 ± 0.036 | 115.2 ± 4.0 | 6.5 ± 0.2 |
| Facebook_c | Adam | 13.417 ± 0.000 | 84.0 ± 0.0 | 0.82 ± 0.09 | 11.942 ± 0.000 | 104.0 ± 0.0 | 0.82 ± 0.08 |
| Facebook_t | Deep | 9.464 ± 0.047 | 198.1 ± 16.3 | 3.1 ± 0.2 | 8.481 ± 0.064 | 222.4 ± 25.1 | 6.8 ± 0.2 |
| Facebook_t | Adam | 10.043 ± 0.000 | 93.0 ± 0.0 | 1.26 ± 0.17 | 9.376 ± 0.000 | 109.0 ± 0.0 | 1.23 ± 0.11 |
| Twitch | Deep | 5.268 ± 0.074 | 4724.5 ± 215.9 | 8.9 ± 0.4 | 0.794 ± 0.013 | 526.7 ± 44.4 | 7.6 ± 1.7 |
| Twitch | Adam | 4.933 ± 0.001 | 3214.9 ± 31.1 | 3.76 ± 0.09 | 0.722 ± 0.000 | 97.0 ± 0.0 | 3.46 ± 0.04 |
| DBLP | Deep | 1.751 ± 0.029 | 535.0 ± 35.4 | 3.7 ± 0.2 | 1.595 ± 0.027 | 554.8 ± 59.6 | 6.2 ± 1.8 |
| DBLP | Adam | 2.109 ± 0.000 | 147.0 ± 0.0 | 1.79 ± 0.02 | 2.000 ± 0.000 | 138.0 ± 0.0 | 1.71 ± 0.04 |
| Pokec-a | Deep | 7.578 ± 0.008 | 69991.8 ± 900.6 | 189.0 ± 6.6 | 6.864 ± 0.007 | 78701.4 ± 1593.2 | 148.6 ± 2.5 |
| Pokec-a | Adam | 9.233 ± 0.0002 | 5470.5 ± 2.0 | 124.90 ± 5.73 | 8.209 ± 0.0001 | 6393.3 ± 2.8 | 96.82 ± 1.52 |
| Pokec-g | Deep | 26.620 ± 0.014 | 115739.4 ± 2470.8 | 186.8 ± 4.5 | 22.093 ± 0.017 | 70893.8 ± 2355.8 | 179.5 ± 3.3 |
| Pokec-g | Adam | 33.833 ± 0.000 | 7839.0 ± 0.0 | 111.42 ± 20.13 | 29.532 ± 0.000 | 6117.0 ± 0.0 | 110.41 ± 3.29 |


