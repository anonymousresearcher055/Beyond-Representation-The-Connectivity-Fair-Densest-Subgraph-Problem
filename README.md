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
## Experimental evaluation

We evaluate the methods on synthetic and real attributed networks. The synthetic experiments isolate the effects of background connectivity and planted edge patterns on density and fairness. The real networks vary in size, group representation, homophily, and cross-group connectivity. We measure Density, Diverse Density, `eFairD`, `eFairDivD`, and selected subgraph size. Results are averaged over 10 runs.

**Jump to:** [Datasets](#datasets) · [Synthetic setup](#synthetic-experimental-setup) · [Extraction results](#densest-subgraph-results) · [Optimality](#solution-quality-relative-to-optimum-values) · [Running times](#running-times) · [Variability](#variability-of-fairness-aware-deep-and-adam-methods)

### Datasets

#### Real-world datasets

We use seven attributed network instances from Deezer, Facebook, Twitch, DBLP, and Pokec. Facebook is evaluated using academic concentration (`Facebook_c`) and education degree (`Facebook_t`), while Pokec is evaluated using age (`Pokec-a`) and gender (`Pokec-g`). For Deezer, Twitch, and DBLP, the groups are defined by gender, streaming language, and ethnicity, respectively.

| Dataset | Attribute | Nodes | Edges | Groups | Avg. diverse degree |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **Deezer** | Gender | 28,281 | 92,752 | 2 | 3.12 |
| **Facebook_c** | Academic concentration | 4,039 | 88,234 | 2 | 4.63 |
| **Facebook_t** | Education degree | 3,578 | 75,993 | 4 | 23.95 |
| **Twitch** | Streaming language | 31,465 | 536,510 | 6 | 0.57 |
| **DBLP** | Ethnicity | 25,944 | 133,788 | 5 | 3.41 |
| **Pokec-a** | Age | 1,632,636 | 22,301,602 | 2 | 4.22 |
| **Pokec-g** | Gender | 1,632,636 | 22,301,602 | 2 | 15.72 |

The group share gives the fraction of nodes belonging to a group. Incident edge share is the fraction of graph edges with at least one endpoint in that group. Cross-group edges are counted for both endpoint groups, so these shares may sum to more than one. Average degree and average diverse degree measure overall and cross-group connectivity, respectively. A relative homophily value above one indicates a greater within-group edge density than the graph-wide baseline.

<details>
<summary><strong>Deezer</strong> — Gender · 2 groups</summary>

| Group | Nodes | Group share | Incident edges | Incident share | Avg. degree | Rel. homophily | Avg. diverse degree |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Red | 12,538 | 0.44 | 61,770 | 0.666 | 6.34 | 0.972 | 3.51 |
| Blue | 15,743 | 0.56 | 75,030 | 0.809 | 6.73 | 1.078 | 2.80 |

</details>

<details>
<summary><strong>Facebook_c</strong> — Academic concentration · 2 groups</summary>

| Group | Nodes | Group share | Incident edges | Incident share | Avg. degree | Rel. homophily | Avg. diverse degree |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Red | 367 | 0.09 | 10,439 | 0.118 | 31.38 | 1.485 | 25.50 |
| Blue | 3,672 | 0.91 | 87,155 | 0.988 | 44.92 | 1.067 | 2.55 |

</details>

<details>
<summary><strong>Facebook_t</strong> — Education degree · 4 groups</summary>

| Group | Nodes | Group share | Incident edges | Incident share | Avg. degree | Rel. homophily | Avg. diverse degree |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No Education | 995 | 0.28 | 25,507 | 0.336 | 29.71 | 0.69 | 21.56 |
| High School | 220 | 0.06 | 7,117 | 0.094 | 33.87 | 1.17 | 30.83 |
| College | 1,472 | 0.41 | 53,418 | 0.703 | 50.68 | 1.65 | 21.90 |
| Graduate | 891 | 0.25 | 32,798 | 0.432 | 45.32 | 1.61 | 28.30 |

</details>

<details>
<summary><strong>Twitch</strong> — Streaming language · 6 groups</summary>

| Group | Nodes | Group share | Incident edges | Incident share | Avg. degree | Rel. homophily | Avg. diverse degree |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DE | 9,259 | 0.29 | 164,314 | 0.306 | 35.11 | 3.46 | 0.39 |
| ES | 5,549 | 0.18 | 89,421 | 0.167 | 31.59 | 5.15 | 0.64 |
| FR | 6,727 | 0.21 | 127,349 | 0.237 | 37.31 | 5.04 | 0.55 |
| PT | 2,454 | 0.08 | 45,950 | 0.086 | 36.23 | 13.17 | 1.22 |
| RU | 4,684 | 0.15 | 48,804 | 0.091 | 20.34 | 3.91 | 0.50 |
| ZH | 2,792 | 0.09 | 69,613 | 0.130 | 49.24 | 16.07 | 0.63 |

</details>

<details>
<summary><strong>DBLP</strong> — Ethnicity · 5 groups</summary>

| Group | Nodes | Group share | Incident edges | Incident share | Avg. degree | Rel. homophily | Avg. diverse degree |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| African | 475 | 0.02 | 4,369 | 0.033 | 9.40 | 2.190 | 8.99 |
| Asian | 15,141 | 0.58 | 98,466 | 0.736 | 10.84 | 1.440 | 2.17 |
| European | 8,394 | 0.32 | 57,993 | 0.433 | 9.60 | 1.615 | 4.21 |
| Hispanic | 852 | 0.03 | 7,627 | 0.057 | 9.51 | 3.303 | 8.39 |
| Middle Eastern/N. African | 1,082 | 0.04 | 9,600 | 0.072 | 9.54 | 3.088 | 8.21 |

</details>

<details>
<summary><strong>Pokec-a</strong> — Age · 2 groups</summary>

| Group | Nodes | Group share | Incident edges | Incident share | Avg. degree | Rel. homophily | Avg. diverse degree |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ≤ 30 | 1,392,851 | 0.85 | 22,111,798 | 0.991 | 29.28 | 1.150 | 2.47 |
| > 30 | 239,785 | 0.15 | 3,635,663 | 0.163 | 15.95 | 0.395 | 14.37 |

</details>

<details>
<summary><strong>Pokec-g</strong> — Gender · 2 groups</summary>

| Group | Nodes | Group share | Incident edges | Incident share | Avg. degree | Rel. homophily | Avg. diverse degree |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Red | 828,301 | 0.51 | 18,129,542 | 0.813 | 28.28 | 0.922 | 15.49 |
| Blue | 804,335 | 0.49 | 17,006,397 | 0.763 | 26.33 | 0.771 | 15.96 |

</details>

### Synthetic experimental setup

We generate random-regular background graphs with two disjoint planted sets, S₁ and S₂, having the same size and group composition. We add edges within each set to create different connectivity profiles. In the two-group `eFairD` setting, S₁ has stronger worst-group incident connectivity, whereas S₂ is denser but concentrates more edges within one group. In the three-group `eFairDivD` setting, S₁ distributes cross-group edges more evenly, whereas S₂ concentrates cross-group edges between two groups. We vary the background degree E₀ from 1 to 10 and generate 10 graphs per configuration.

| Parameter | S₁ | S₂ |
| :--- | ---: | ---: |
| **Shared parameters** | | |
| Nodes (n) | 1,000 | 1,000 |
| Background degree (E₀) | 5 | 5 |
| Planted-set size (s) | 100 | 100 |
| Added edges (Mₛ) | 1,000 | 1,400 |
| **eFairD: two groups** | | |
| Red fraction (global, planted) | (0.5, 0.5) | (0.5, 0.5) |
| Edge-type proportions (RR, RB, BB) | (0.30, 0.40, 0.30) | (0.70, 0.05, 0.25) |
| **eFairDivD: three groups** | | |
| Group fraction (global, planted) | (0.33, 0.33) | (0.33, 0.33) |
| Same-group proportions (RR, BB, YY) | (0.083, 0.083, 0.083) | (0.05, 0.05, 0.05) |
| Cross-group proportions (RB, RY, BY) | (0.25, 0.25, 0.25) | (0.75, 0.05, 0.05) |

### Experiments

#### Densest subgraph results

We compare the methods using Density, `eFairD`, `eFairDivD`, Diverse Density, and the size of the selected subgraph. Results are means over 10 runs. Bold values identify the best observed score for each metric and dataset.

#### Facebook_t, Twitch, DBLP, and Pokec-a

##### Facebook_t

| Method | Density | eFairD | eFairDivD | Diverse Density | Subgraph Size |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Greedy | **70.554** | 1.694 | 1.694 | 34.855 | 186 |
| Spectral | **70.554** | 1.694 | 1.694 | 34.855 | 186 |
| Deep | **70.554** | 1.694 | 1.694 | 34.855 | 186 |
| Adam | **70.554** | 1.694 | 1.694 | 34.855 | 186 |
| GreedyGroupFairness | 34.755 | 9.364 | 8.437 | 21.112 | 286 |
| DeepGroupFairness | 28.914 | 9.464 | 8.421 | 20.126 | 198.1 |
| AdamGroupFairness | 30.022 | **10.043** | 9.118 | 22.667 | 93 |
| GreedyDiverse | 65.120 | 1.785 | 1.785 | 35.127 | 158 |
| DeepDiverse | 56.748 | 4.320 | 4.154 | 32.148 | 522.3 |
| AdamDiverse | 69.449 | 1.719 | 1.719 | **35.399** | 178 |
| GreedyGroupDiverseFairness | 33.619 | 9.463 | 8.696 | 21.331 | 257 |
| DeepGroupDiverseFairness | 33.974 | 9.381 | 8.481 | 22.191 | 222.4 |
| AdamGroupDiverseFairness | 35.156 | 10.165 | **9.376** | 24.982 | 109 |
| DDSP | 38.059 | 2.937 | 2.346 | 18.803 | 492 |
| ColApproxGroup | 47.325 | 1.911 | 1.856 | 25.310 | 375.1 |
| ColApproxDiv | 47.485 | 1.910 | 1.879 | 25.810 | 364.4 |

##### Twitch

| Method | Density | eFairD | eFairDivD | Diverse Density | Subgraph Size |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Greedy | **48.809** | 0.000 | 0.000 | 0.000 | 513 |
| Spectral | 40.273 | 0.000 | 0.000 | 0.000 | 651 |
| Deep | 40.318 | 0.000 | 0.000 | 0.502 | 2589.6 |
| Adam | 43.943 | 0.000 | 0.000 | 0.065 | 880 |
| GreedyGroupFairness | 35.489 | 3.919 | 0.205 | 0.860 | 3633 |
| DeepGroupFairness | 33.889 | **5.268** | 0.156 | 0.702 | 4724.5 |
| AdamGroupFairness | 34.779 | 4.933 | 0.172 | 0.730 | 3214.9 |
| GreedyDiverse | 6.900 | 0.000 | 0.000 | **3.775** | 40 |
| DeepDiverse | 10.667 | 0.914 | 0.607 | 3.193 | 532.1 |
| AdamDiverse | 5.611 | 0.000 | 0.000 | 2.500 | 36 |
| GreedyGroupDiverseFairness | 7.502 | 1.393 | **0.961** | 3.081 | 285 |
| DeepGroupDiverseFairness | 9.893 | 1.455 | 0.794 | 3.107 | 526.7 |
| AdamGroupDiverseFairness | 4.526 | 0.814 | 0.722 | 2.680 | 97 |
| DDSP | 37.128 | 4.094 | 0.149 | 0.649 | 3090 |
| ColApproxGroup | 26.275 | 1.520 | 0.101 | 0.424 | 3037.4 |
| ColApproxDiv | 7.359 | 0.930 | 0.598 | 2.809 | 390.4 |

##### DBLP

| Method | Density | eFairD | eFairDivD | Diverse Density | Subgraph Size |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Greedy | **21.500** | 0.000 | 0.000 | 5.886 | 44 |
| Spectral | **21.500** | 0.000 | 0.000 | 5.886 | 44 |
| Deep | 14.671 | 0.349 | 0.344 | 5.994 | 612.6 |
| Adam | 19.408 | 0.000 | 0.000 | 5.327 | 49 |
| GreedyGroupFairness | 11.641 | 1.846 | 1.786 | 6.128 | 117 |
| DeepGroupFairness | 7.864 | 1.751 | 1.605 | 5.293 | 535 |
| AdamGroupFairness | 8.551 | **2.109** | 1.993 | 5.952 | 147 |
| GreedyDiverse | 17.500 | 0.000 | 0.000 | **10.361** | 36 |
| DeepDiverse | 12.402 | 0.814 | 0.792 | 7.697 | 858.7 |
| AdamDiverse | 11.566 | 0.781 | 0.774 | 7.806 | 279 |
| GreedyGroupDiverseFairness | 10.663 | 1.683 | 1.634 | 6.874 | 413 |
| DeepGroupDiverseFairness | 8.239 | 1.682 | 1.595 | 5.554 | 554.8 |
| AdamGroupDiverseFairness | 9.239 | 2.087 | **2.000** | 6.500 | 138 |
| DDSP | 9.714 | 1.000 | 0.951 | 4.222 | 185 |
| ColApproxGroup | 11.177 | 0.437 | 0.430 | 5.856 | 1210.1 |
| ColApproxDiv | 13.043 | 0.737 | 0.728 | 8.039 | 588.8 |

##### Pokec-a

| Method | Density | eFairD | eFairDivD | Diverse Density | Subgraph Size |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Greedy | **41.109** | 4.770 | 4.614 | 4.614 | 8032 |
| Spectral | 27.661 | 3.358 | 3.242 | 3.242 | 248973 |
| Deep | 32.494 | 3.801 | 3.674 | 3.674 | 144479.7 |
| Adam | 40.054 | 4.575 | 4.434 | 4.434 | 8857 |
| GreedyGroupFairness | 24.376 | **10.046** | 8.835 | 8.835 | 4971 |
| DeepGroupFairness | 18.591 | 7.578 | 6.635 | 6.635 | 69991.8 |
| AdamGroupFairness | 24.820 | 9.233 | 8.126 | 8.126 | 5470.5 |
| GreedyDiverse | 26.857 | 9.964 | 9.000 | **9.000** | 5416 |
| DeepDiverse | 20.489 | 7.593 | 6.847 | 6.847 | 81097.5 |
| AdamDiverse | 27.135 | 9.030 | 8.209 | 8.209 | 6390 |
| GreedyGroupDiverseFairness | 26.857 | 9.964 | **9.000** | 9.000 | 5416 |
| DeepGroupDiverseFairness | 20.565 | 7.613 | 6.864 | 6.864 | 78701.4 |
| AdamGroupDiverseFairness | 27.141 | 9.029 | 8.209 | 8.209 | 6393.3 |
| Balance Spectral | 13.391 | 2.964 | 2.570 | 2.570 | 479482 |
| DDSP | 27.165 | 7.774 | 6.754 | 6.754 | 15482 |
| ColApproxGroup | 33.664 | 5.851 | 5.565 | 5.565 | 66247.4 |
| ColApproxDiv | 23.677 | 8.284 | 7.564 | 7.564 | 45554 |

On Facebook_t, `AdamDiverse` achieves a Diverse Density of 35.399 but an `eFairDivD` of only 1.719. In contrast, `AdamGroupDiverseFairness` achieves an `eFairDivD` of 9.376. On DBLP, Greedy maximizes overall density among the reported methods but achieves zero under both connectivity-fairness measures, while the fairness-aware methods identify subgraphs with positive worst-group connectivity. These differences illustrate why overall density and overall diverse density do not determine connectivity fairness.

#### Deezer, Facebook_c, and Pokec-g

<details>
<summary><strong>Deezer</strong></summary>

| Method | Density | eFairD | eFairDivD | Diverse Density | Subgraph Size |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Greedy | **8.463** | 5.423 | 4.357 | 4.357 | 272 |
| Spectral | 8.204 | 4.981 | 4.074 | 4.074 | 54 |
| Deep | 7.435 | 4.361 | 3.440 | 3.440 | 1056.3 |
| Adam | 7.746 | 4.115 | 3.445 | 3.445 | 282 |
| GreedyGroupFairness | 7.844 | **6.275** | 4.749 | 4.749 | 167 |
| DeepGroupFairness | 6.858 | 5.123 | 3.666 | 3.666 | 1189.7 |
| AdamGroupFairness | 7.653 | 5.785 | 4.450 | 4.450 | 242 |
| GreedyDiverse | 7.504 | 5.863 | 4.838 | 4.838 | 117 |
| DeepDiverse | 6.662 | 4.889 | 3.904 | 3.904 | 1015.4 |
| AdamDiverse | 7.689 | 6.000 | 4.874 | **4.874** | 183 |
| GreedyGroupDiverseFairness | 7.917 | 6.005 | 4.850 | 4.850 | 206 |
| DeepGroupDiverseFairness | 6.675 | 4.894 | 3.905 | 3.905 | 1041.2 |
| AdamGroupDiverseFairness | 7.689 | 6.000 | **4.874** | 4.874 | 183 |
| Balance Spectral | 3.429 | 2.197 | 1.685 | 1.685 | 3898 |
| DDSP | 8.319 | 5.798 | 4.424 | 4.424 | 382 |
| ColApproxGroup | 7.108 | 4.599 | 3.622 | 3.622 | 1343.4 |
| ColApproxDiv | 7.054 | 5.118 | 4.099 | 4.099 | 1075 |

</details>

<details>
<summary><strong>Facebook_c</strong></summary>

| Method | Density | eFairD | eFairDivD | Diverse Density | Subgraph Size |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Greedy | **77.347** | 2.257 | 2.243 | 2.243 | 202 |
| Spectral | **77.347** | 2.257 | 2.243 | 2.243 | 202 |
| Deep | 76.731 | 2.254 | 2.239 | 2.239 | 201 |
| Adam | **77.347** | 2.257 | 2.243 | 2.243 | 202 |
| GreedyGroupFairness | 31.634 | **13.512** | 11.780 | 11.780 | 82 |
| DeepGroupFairness | 30.628 | 12.130 | 10.251 | 10.251 | 152 |
| AdamGroupFairness | 33.036 | 13.417 | 11.762 | 11.762 | 84 |
| GreedyDiverse | 40.784 | 13.250 | 12.000 | **12.000** | 116 |
| DeepDiverse | 37.156 | 12.561 | 11.213 | 11.213 | 121.4 |
| AdamDiverse | 39.106 | 13.250 | 11.942 | 11.942 | 104 |
| GreedyGroupDiverseFairness | 40.784 | 13.250 | **12.000** | 12.000 | 116 |
| DeepGroupDiverseFairness | 39.000 | 13.064 | 11.748 | 11.748 | 115.2 |
| AdamGroupDiverseFairness | 39.106 | 13.250 | 11.942 | 11.942 | 104 |
| Balance Spectral | 37.614 | 2.431 | 1.474 | 1.474 | 466 |
| DDSP | 41.553 | 3.442 | 1.445 | 1.445 | 398 |
| ColApproxGroup | 63.039 | 3.950 | 3.595 | 3.595 | 270.2 |
| ColApproxDiv | 40.784 | 13.250 | **12.000** | 12.000 | 116 |

</details>

<details>
<summary><strong>Pokec-g</strong></summary>

| Method | Density | eFairD | eFairDivD | Diverse Density | Subgraph Size |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Greedy | **41.109** | 33.028 | 27.999 | 27.999 | 8032 |
| Spectral | 27.661 | 21.156 | 16.556 | 16.556 | 248973 |
| Deep | 32.494 | 25.842 | 20.259 | 20.259 | 144156.9 |
| Adam | 40.054 | 32.557 | 27.953 | 27.953 | 8857 |
| GreedyGroupFairness | 38.897 | **34.444** | 29.991 | 29.991 | 6944 |
| DeepGroupFairness | 31.904 | 26.620 | 21.385 | 21.385 | 115739.4 |
| AdamGroupFairness | 38.844 | 33.833 | 29.138 | 29.138 | 7839 |
| GreedyDiverse | 38.322 | 33.772 | 30.310 | **30.310** | 6070 |
| DeepDiverse | 29.479 | 25.390 | 22.094 | 22.094 | 72252.4 |
| AdamDiverse | 36.744 | 32.656 | 29.532 | 29.532 | 6117 |
| GreedyGroupDiverseFairness | 38.392 | 33.789 | **30.310** | 30.310 | 6108 |
| DeepGroupDiverseFairness | 29.448 | 25.371 | 22.093 | 22.093 | 70893.8 |
| AdamGroupDiverseFairness | 36.744 | 32.656 | 29.532 | 29.532 | 6117 |
| Balance Spectral | 27.485 | 21.647 | 16.669 | 16.669 | 210710 |
| DDSP | 40.999 | 33.465 | 27.878 | 27.878 | 8924 |
| ColApproxGroup | 33.796 | 27.466 | 22.327 | 22.327 | 63264.9 |
| ColApproxDiv | 32.910 | 27.930 | 23.810 | 23.810 | 53904 |

</details>

#### Solution quality relative to optimum values

We assess solution quality relative to exact optima when available and certified upper bounds otherwise. `UB` denotes an upper bound on the optimal value, and `Opt.` denotes an exact optimum. Each method entry gives its achieved score and the corresponding percentage. Where no exact optimum is available, the percentage relative to `UB` provides a lower bound on the fraction of the optimum achieved. A dash denotes an unavailable exact optimum.

##### eFairD

| Dataset | UB | Opt. | Greedy | Deep | Adam |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Deezer | 6.806 | 6.410 | **6.275 (97.9%)** | 5.123 (79.9%) | 5.785 (90.3%) |
| Facebook_c | 13.512 | 13.512 | **13.512 (100.0%)** | 12.130 (89.8%) | 13.417 (99.3%) |
| Facebook_t | 11.197 | 10.648 | 9.364 (87.9%) | 9.464 (88.9%) | **10.043 (94.3%)** |
| Twitch | 8.770 | -- | 3.919 (44.7%) | **5.268 (60.1%)** | 4.933 (56.2%) |
| DBLP | 3.286 | 2.342 | 1.846 (78.8%) | 1.751 (74.8%) | **2.109 (90.0%)** |
| Pokec-a | 10.083 | -- | **10.046 (99.6%)** | 7.578 (75.2%) | 9.233 (91.6%) |
| Pokec-g | 35.179 | -- | **34.444 (97.9%)** | 26.620 (75.7%) | 33.833 (96.2%) |

##### eFairDivD

| Dataset | UB | Opt. | Greedy | Deep | Adam |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Deezer | 5.000 | 5.000 | 4.850 (97.0%) | 3.905 (78.1%) | **4.874 (97.5%)** |
| Facebook_c | 12.000 | 12.000 | **12.000 (100.0%)** | 11.748 (97.9%) | 11.942 (99.5%) |
| Facebook_t | 10.069 | 9.775 | 8.696 (89.0%) | 8.481 (86.8%) | **9.376 (95.9%)** |
| Twitch | 1.269 | 1.056 | **0.961 (91.0%)** | 0.794 (75.2%) | 0.722 (68.3%) |
| DBLP | 2.857 | 2.187 | 1.634 (74.7%) | 1.595 (72.9%) | **2.000 (91.4%)** |
| Pokec-a | 9.027 | -- | **9.000 (99.7%)** | 6.864 (76.0%) | 8.209 (90.9%) |
| Pokec-g | 30.441 | -- | **30.310 (99.6%)** | 22.093 (72.6%) | 29.532 (97.0%) |

##### Diverse Density

| Dataset | Opt. | Greedy | Deep | Adam |
| :--- | ---: | ---: | ---: | ---: |
| Deezer | 5.000 | 4.838 (96.8%) | 3.904 (78.1%) | **4.874 (97.5%)** |
| Facebook_c | 12.000 | **12.000 (100.0%)** | 11.213 (93.4%) | 11.942 (99.5%) |
| Facebook_t | 35.462 | 35.127 (99.1%) | 32.148 (90.7%) | **35.399 (99.8%)** |
| Twitch | 3.808 | **3.775 (99.1%)** | 3.193 (83.8%) | 2.500 (65.7%) |
| DBLP | 10.361 | **10.361 (100.0%)** | 7.697 (74.3%) | 7.806 (75.3%) |
| Pokec-a | 9.027 | **9.000 (99.7%)** | 6.847 (75.9%) | 8.209 (90.9%) |
| Pokec-g | 30.441 | **30.310 (99.6%)** | 22.094 (72.6%) | 29.532 (97.0%) |

### Efficiency and variability

We measure execution time and variability over 10 runs on a system with **128 CPU threads, 1 TiB RAM, and two NVIDIA H200 GPUs**. Running times are in seconds, with values given as **mean ± standard deviation**. A dash indicates that the corresponding method was not evaluated on that dataset.

#### Running times

The comparison covers both smaller attributed networks and the large Pokec network.

##### Small and medium-sized networks

| Method | Deezer | Facebook_c | Facebook_t | Twitch | DBLP |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Greedy | 0.113 ± 0.012 | 0.056 ± 0.001 | 0.048 ± 0.001 | 0.628 ± 0.021 | 0.136 ± 0.002 |
| GreedyGroupFairness | 1.071 ± 0.004 | 0.571 ± 0.041 | 0.617 ± 0.024 | 8.042 ± 0.115 | 2.090 ± 0.046 |
| GreedyDiverse | 0.460 ± 0.046 | 0.076 ± 0.012 | 0.209 ± 0.018 | 0.313 ± 0.043 | 0.410 ± 0.028 |
| GreedyGroupDiverseFairness | 1.104 ± 0.056 | 0.557 ± 0.034 | 0.620 ± 0.025 | 12.20 ± 0.34 | 2.110 ± 0.038 |
| Spectral | 0.201 ± 0.002 | 0.0622 ± 0.0004 | 0.056 ± 0.012 | 0.610 ± 0.042 | 0.212 ± 0.002 |
| Deep | 17.16 ± 7.28 | 24.51 ± 8.26 | 27.49 ± 6.34 | 22.58 ± 9.38 | 27.93 ± 6.40 |
| DeepGroupFairness | 6.93 ± 0.44 | 6.39 ± 0.15 | 3.15 ± 0.21 | 8.90 ± 0.40 | 3.68 ± 0.25 |
| DeepDiverse | 28.02 ± 5.72 | 17.00 ± 7.39 | 25.52 ± 7.66 | 20.40 ± 8.76 | 27.89 ± 6.06 |
| DeepGroupDiverseFairness | 2.87 ± 0.15 | 6.51 ± 0.20 | 6.81 ± 0.21 | 7.60 ± 1.72 | 6.17 ± 1.80 |
| Adam | 0.787 ± 0.272 | 0.584 ± 0.038 | 0.524 ± 0.051 | 2.193 ± 0.078 | 0.715 ± 0.018 |
| AdamGroupFairness | 1.279 ± 0.294 | 0.816 ± 0.086 | 1.261 ± 0.173 | 3.757 ± 0.092 | 1.790 ± 0.020 |
| AdamDiverse | 0.680 ± 0.034 | 0.509 ± 0.049 | 0.528 ± 0.022 | 0.700 ± 0.038 | 0.651 ± 0.023 |
| AdamGroupDiverseFairness | 0.989 ± 0.081 | 0.823 ± 0.075 | 1.226 ± 0.113 | 3.464 ± 0.044 | 1.705 ± 0.036 |
| Balance Spectral | 0.395 ± 0.083 | 0.045 ± 0.001 | -- | -- | -- |
| DDSP | 13.37 ± 0.21 | 0.347 ± 0.035 | 0.497 ± 0.019 | 39.62 ± 0.47 | 10.58 ± 0.22 |
| ColApproxGroup | 0.574 ± 0.020 | 0.265 ± 0.006 | 0.286 ± 0.008 | 5.21 ± 0.12 | 1.14 ± 0.05 |
| ColApproxDiv | 0.228 ± 0.001 | 0.065 ± 0.015 | 0.193 ± 0.018 | 0.327 ± 0.069 | 0.247 ± 0.029 |

##### Pokec large-scale networks

| Method | Pokec-a | Pokec-g |
| :--- | ---: | ---: |
| Greedy | 63.08 ± 0.36 | 64.88 ± 4.07 |
| GreedyGroupFairness | 456.93 ± 4.19 | 476.66 ± 4.80 |
| GreedyDiverse | 94.71 ± 2.80 | 220.28 ± 4.29 |
| GreedyGroupDiverseFairness | 470.56 ± 6.93 | 459.66 ± 4.63 |
| Spectral | 123.48 ± 6.97 | 126.00 ± 3.10 |
| Deep | 385.51 ± 9.25 | 384.43 ± 12.86 |
| DeepGroupFairness | 188.99 ± 6.56 | 186.83 ± 4.50 |
| DeepDiverse | 114.53 ± 5.16 | 223.23 ± 7.75 |
| DeepGroupDiverseFairness | 148.63 ± 2.51 | 179.55 ± 3.30 |
| Adam | 80.84 ± 9.62 | 85.35 ± 4.11 |
| AdamGroupFairness | 124.90 ± 5.73 | 111.42 ± 20.13 |
| AdamDiverse | 16.96 ± 0.17 | 19.88 ± 0.12 |
| AdamGroupDiverseFairness | 96.82 ± 1.52 | 110.41 ± 3.29 |
| Balance Spectral | 23.97 ± 0.60 | 21.85 ± 0.40 |
| DDSP | -- | -- |
| ColApproxGroup | 4081.62 ± 500.83 | 3290.56 ± 40.88 |
| ColApproxDiv | 55.86 ± 9.42 | 120.60 ± 12.87 |

#### Variability of fairness-aware Deep and Adam methods

We examine the variability of the Deep and Adam fairness-aware methods in the objective score, selected subgraph size, and running time.

##### eFairD

| Dataset | Method | Score | Subgraph size | Time (s) |
| :--- | ---: | ---: | ---: | ---: |
| Deezer | Deep | 5.123 ± 0.008 | 1189.7 ± 26.4 | 6.9 ± 0.4 |
| Deezer | Adam | 5.785 ± 0.000 | 242.0 ± 0.0 | 1.28 ± 0.29 |
| Facebook_c | Deep | 12.130 ± 0.142 | 152.0 ± 7.5 | 6.4 ± 0.2 |
| Facebook_c | Adam | 13.417 ± 0.000 | 84.0 ± 0.0 | 0.82 ± 0.09 |
| Facebook_t | Deep | 9.464 ± 0.047 | 198.1 ± 16.3 | 3.1 ± 0.2 |
| Facebook_t | Adam | 10.043 ± 0.000 | 93.0 ± 0.0 | 1.26 ± 0.17 |
| Twitch | Deep | 5.268 ± 0.074 | 4724.5 ± 215.9 | 8.9 ± 0.4 |
| Twitch | Adam | 4.933 ± 0.001 | 3214.9 ± 31.1 | 3.76 ± 0.09 |
| DBLP | Deep | 1.751 ± 0.029 | 535.0 ± 35.4 | 3.7 ± 0.2 |
| DBLP | Adam | 2.109 ± 0.000 | 147.0 ± 0.0 | 1.79 ± 0.02 |
| Pokec-a | Deep | 7.578 ± 0.008 | 69991.8 ± 900.6 | 189.0 ± 6.6 |
| Pokec-a | Adam | 9.233 ± 0.0002 | 5470.5 ± 2.0 | 124.90 ± 5.73 |
| Pokec-g | Deep | 26.620 ± 0.014 | 115739.4 ± 2470.8 | 186.8 ± 4.5 |
| Pokec-g | Adam | 33.833 ± 0.000 | 7839.0 ± 0.0 | 111.42 ± 20.13 |

##### eFairDivD

| Dataset | Method | Score | Subgraph size | Time (s) |
| :--- | ---: | ---: | ---: | ---: |
| Deezer | Deep | 3.905 ± 0.007 | 1041.2 ± 31.6 | 2.9 ± 0.1 |
| Deezer | Adam | 4.874 ± 0.000 | 183.0 ± 0.0 | 0.99 ± 0.08 |
| Facebook_c | Deep | 11.748 ± 0.036 | 115.2 ± 4.0 | 6.5 ± 0.2 |
| Facebook_c | Adam | 11.942 ± 0.000 | 104.0 ± 0.0 | 0.82 ± 0.08 |
| Facebook_t | Deep | 8.481 ± 0.064 | 222.4 ± 25.1 | 6.8 ± 0.2 |
| Facebook_t | Adam | 9.376 ± 0.000 | 109.0 ± 0.0 | 1.23 ± 0.11 |
| Twitch | Deep | 0.794 ± 0.013 | 526.7 ± 44.4 | 7.6 ± 1.7 |
| Twitch | Adam | 0.722 ± 0.000 | 97.0 ± 0.0 | 3.46 ± 0.04 |
| DBLP | Deep | 1.595 ± 0.027 | 554.8 ± 59.6 | 6.2 ± 1.8 |
| DBLP | Adam | 2.000 ± 0.000 | 138.0 ± 0.0 | 1.71 ± 0.04 |
| Pokec-a | Deep | 6.864 ± 0.007 | 78701.4 ± 1593.2 | 148.6 ± 2.5 |
| Pokec-a | Adam | 8.209 ± 0.0001 | 6393.3 ± 2.8 | 96.82 ± 1.52 |
| Pokec-g | Deep | 22.093 ± 0.017 | 70893.8 ± 2355.8 | 179.5 ± 3.3 |
| Pokec-g | Adam | 29.532 ± 0.000 | 6117.0 ± 0.0 | 110.41 ± 3.29 |


