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

