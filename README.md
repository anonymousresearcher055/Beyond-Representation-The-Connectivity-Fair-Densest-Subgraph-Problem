<div align="center">

# Beyond Representation: The Connectivity-Fair Densest Subgraph Problem

**Research code and additional experimental results for the paper.**

<a href="#quick-start">Quick start</a> ·
<a href="#objectives">Objectives</a> ·
<a href="#implemented-methods">Methods</a> ·
<a href="#input-datasets">Datasets</a> ·
<a href="#experimental-evaluation">Experimental results</a> ·
<a href="#reproducibility-and-data-notes">Reproducibility</a>

<br>

![Python](https://img.shields.io/badge/Python-research%20code-3776AB)
![NetworkX](https://img.shields.io/badge/NetworkX-graph%20algorithms-376795)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20%26%20Adam-FF6F00)

</div>

---

This repository accompanies the manuscript **Beyond Representation: The Connectivity-Fair Densest Subgraph Problem**, submitted to **The Web Conference 2027** (Responsible Web).

A dense subgraph can contain similar numbers of nodes from different groups while most of its edges involve only some of those groups. We study **connectivity fairness** in densest subgraph discovery: how to find a dense subgraph without leaving any group poorly connected. We introduce **edge fair density (eFairD)** and **edge fair diverse density (eFairDivD)**, two max-min objectives that measure the weakest group's connectivity.

## Repository overview

The repository provides:

- Greedy, GCN-based Deep, and direct Adam methods for density, Diverse Density, eFairD, and eFairDivD.
- A single `main.py` that runs selected methods with **editable global parameters**, without command-line arguments.
- An included DBLP coauthorship graph and a synthetic example with planted subgraphs.
- Automatic graph normalization, computation of connectivity-fairness metrics, and exported selected subgraphs.
- The main experimental results from the manuscript **and additional real-world, optimality, runtime, and variability results**.

The full experimental evaluation covers more datasets and baselines than are distributed here. The [reproducibility notes](#reproducibility-and-data-notes) explain that distinction.

## Quick start

### Installation

Clone the repository and create a Python environment. Python **3.10 or newer** is a practical starting point, but the repository does not currently document a fully tested version combination.

```bash
git clone https://github.com/anonymousresearcher055/Beyond-Representation-The-Connectivity-Fair-Densest-Subgraph-Problem.git
cd Beyond-Representation-The-Connectivity-Fair-Densest-Subgraph-Problem

python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# On Windows: .venv\Scripts\activate

python -m pip install --upgrade pip
python -m pip install numpy pandas scipy networkx scikit-learn matplotlib tensorflow
```

The greedy methods use NetworkX and the scientific Python packages. **Deep and Adam require TensorFlow**. GPU acceleration is optional, subject to a compatible TensorFlow installation and hardware environment. The repository does not currently contain a `requirements.txt` or pinned dependency versions, so the installation above is guidance based on the source imports, not a verified environment lockfile.

### Run the included experiments

**All experiment settings are global variables at the top of [`main.py`](main.py). There are no command-line flags.**

For a short first run, edit the configuration to run only the four greedy methods on DBLP:

```python
DATASETS_TO_RUN = ["dblp_ethnicity"]
METHODS_TO_RUN = [
    "greedy",
    "greedy_diverse",
    "maxmin_greedy",
    "maxmin_diverse_greedy",
]
N_RUNS_PER_METHOD = 1
PREPARE_ONLY = False
```

Then run from the repository root:

```bash
python main.py
```

For **all 12 methods on DBLP**, set:

```python
DATASETS_TO_RUN = ["dblp_ethnicity"]
METHODS_TO_RUN = None
```

For **only the fairness-aware eFairD and eFairDivD methods**, set:

```python
METHODS_TO_RUN = [
    "maxmin_greedy",
    "deepGroupMaxMin",
    "adamGroupMaxMin",
    "maxmin_diverse_greedy",
    "deepGroupDiverseMaxMin",
    "adamGroupDiverseMaxMin",
]
```

To run the supplied synthetic example, set `DATASETS_TO_RUN` to its complete folder name:

```python
DATASETS_TO_RUN = [
    "n1000__E05__g0_0.34_g1_0.33_g2_0.33__base_unbiased__seed12__run5"
]
```

Using `DATASETS_TO_RUN = None` selects **every folder** in `datasets/`, while `METHODS_TO_RUN = None` selects **every implemented method**. Note that running all methods, particularly on DBLP or larger graphs, can be computationally expensive.

### Global configuration

| Variable | Current default | Meaning |
| :--- | :--- | :--- |
| `STest` | `"S1"` | Target planted set used for synthetic ground-truth comparisons |
| `root_name` | `"datasets"` | Directory name containing dataset folders |
| `DATASETS_ROOT` | `PROJECT_ROOT / root_name` | Input dataset root |
| `OUT_ROOT` | `PROJECT_ROOT / "synthetic_results" / root_name` | Root for normalized inputs and selected subgraphs |
| `SUMMARY_CSV` | `PROJECT_ROOT / f"{root_name}_results_{STest}.csv"` | Aggregated result CSV |
| `DATASETS_TO_RUN` | `["dblp_ethnicity"]` | Folders to process, or `None` for all |
| `METHODS_TO_RUN` | `None` | Method identifiers to run, or `None` for all 12 |
| `N_RUNS_PER_METHOD` | `1` | Number of independent invocations per method |
| `TRAINING_EPOCHS` | `1000` | Training epochs for Deep and Adam variants |
| `PREPARE_ONLY` | `False` | When `True`, prepare normalized inputs without running the algorithms |

For a ten-run experiment, set `N_RUNS_PER_METHOD = 10`. To change training length, edit `TRAINING_EPOCHS`. **The default one-run execution is not the ten-run protocol used to produce the reported paper tables.** The code records successful and failed runs, and stochastic outcomes may differ between executions.

## Objectives

Let **G = (V, E)** be an undirected graph whose nodes belong to groups g₁, …, gₖ. For any nonempty selected node set **S**, let **E(S)** be its induced edges.

| Measure | Definition | What it counts |
| :--- | :--- | :--- |
| **Density** | D(S) = &vert;E(S)&vert; / &vert;S&vert; | All induced edges |
| **Group density** | D<sub>i</sub>(S) = &vert;E<sub>i</sub>(S)&vert; / &vert;S&vert; | Induced edges incident to group *i*, including within-group edges |
| **Diverse Density** | D<sup>div</sup>(S) = &vert;E<sup>div</sup>(S)&vert; / &vert;S&vert; | All induced cross-group edges |
| **Group-diverse density** | D<sub>i</sub><sup>div</sup>(S) = &vert;E<sub>i</sub><sup>div</sup>(S)&vert; / &vert;S&vert; | Induced cross-group edges incident to group *i* |

The two proposed optimization objectives maximize the minimum group-specific density.

**Edge fair density (eFairD)**

eFairD(S) = min<sub>1 ≤ i ≤ k</sub> D<sub>i</sub>(S)

**Edge fair diverse density (eFairDivD)**

eFairDivD(S) = min<sub>1 ≤ i ≤ k</sub> D<sub>i</sub><sup>div</sup>(S)

An edge between two different groups contributes to the relevant measure for both endpoint groups. An edge within a group contributes to its group density, but not to its group-diverse density. **All four measures divide by the total number of selected nodes**, not the number of nodes in an individual group.

eFairD accounts for a group's connections both within and across groups. eFairDivD focuses on its connections to *other* groups. Overall Diverse Density is a comparison objective, not a third proposed fairness problem. With exactly two groups, eFairDivD equals overall Diverse Density for every selected subgraph.

These measures capture structural properties of a graph. They do not, on their own, guarantee fairness of downstream decisions or real-world outcomes.

## Implemented methods

The repository contains **12 executable methods**, organized into three families. The fairness-aware variants target the minimum group-specific score.

| Target objective | Greedy peeling | GCN-based Deep | Direct Adam |
| :--- | :--- | :--- | :--- |
| Density | `greedy` (Greedy) | `deep` (Deep) | `adam` (Adam) |
| Diverse Density | `greedy_diverse` (GreedyDiverse) | `deepDiverse` (DeepDiverse) | `adamDiverse` (AdamDiverse) |
| eFairD | `maxmin_greedy` (GreedyGroupFairness) | `deepGroupMaxMin` (DeepGroupFairness) | `adamGroupMaxMin` (AdamGroupFairness) |
| eFairDivD | `maxmin_diverse_greedy` (GreedyGroupDiverseFairness) | `deepGroupDiverseMaxMin` (DeepGroupDiverseFairness) | `adamGroupDiverseMaxMin` (AdamGroupDiverseFairness) |

- **Greedy:** iteratively removes nodes, with the fairness-aware variants guided by the current least-connected group, and retains the best intermediate subgraph.
- **Deep:** uses a graph convolutional network (GCN) to learn node scores from a continuous objective.
- **Adam:** directly optimizes trainable node scores with Adam, without a GCN encoder. Deep also uses Adam as an optimizer. The distinction between the two families is **how node scores are represented**, not simply the optimizer name.

The gradient-based methods rank nodes by learned scores and evaluate selected prefixes under the discrete objective. The fairness-aware methods are heuristics, with no general approximation guarantee asserted here.

The *paper* also reports results for **Spectral, Balance Spectral, DDSP, ColApproxGroup, and ColApproxDiv**, but those methods are **not executable from the current `main.py`**. Their experimental results are retained below for comparison.

## Repository layout

```text
.
├── main.py
├── greedy_algortihm.py
├── Deep Dense subgraph/
│   ├── deepDenseSubgraph.py
│   └── tools/
│       ├── deep_densest_models.py
│       ├── gcn.py
│       ├── metrics.py
│       └── utils.py
├── Adam Dense subgraph/
│   └── AdamDenseSubgraph.py
├── datasets/
│   ├── dblp_ethnicity/
│   │   ├── dblp_ethnicity.csv
│   │   └── dblp_ethnicity.edgelist
│   └── n1000__E05__g0_0.34_g1_0.33_g2_0.33__base_unbiased__seed12__run5/
│       ├── <dataset-name>.edgelist
│       ├── <dataset-name>_attributes.csv
│       ├── planted_subgraphs.csv
│       └── <additional example metadata>
└── README.md
```

`main.py` loads and normalizes input graphs, calls the selected methods, evaluates the returned subgraphs, and saves results. `greedy_algortihm.py` contains the four greedy routines. The Deep and Adam folders contain the respective optimization implementations. The synthetic example also includes files recording its generation parameters and graph statistics.

The layout above omits Python cache artifacts. Folder and source-file names reflect the current repository, including the spelling of `greedy_algortihm.py`.

## Input datasets

The public repository currently distributes **DBLP** and **one synthetic example**. Other datasets studied in the paper are not included in this repository.

Each dataset is stored in a folder under `datasets/`. The main runner identifies the edge list and attribute CSV from the supported filenames or file extensions.

### Edge lists

The code accepts either an unheaded, whitespace-separated two-column file:

```text
0 1
1 2
2 3
```

or a CSV whose header begins with `source,target` or `u,v`. For example, the included DBLP edge list starts with:

```csv
source,target,time
11,108,1996
11,2225,1996
```

The current runner uses only the first two columns of the DBLP file. It ignores timestamps when constructing the undirected coauthorship graph. Self-loops are removed and repeated unordered node pairs are collapsed to a single edge.

### Node-group attributes

The synthetic example uses numeric groups:

```csv
node,attribute
0,0
1,2
2,0
```

DBLP uses string-valued groups:

```csv
node_id,ethnicity
1,asian
2,asian
3,asian
```

The input loader recognizes `node_id`, `node`, and `node_idx` as node-identifier columns, and `attribute` or `ethnicity` as the group column. All edge endpoints must have attributes. No manual DBLP conversion is required.

### Optional planted ground truth

Synthetic datasets can include `planted_subgraphs.csv` with a selected-set name and node ID, for example:

```csv
set_name,node
S1,1
S1,3
S2,14
```

The main runner uses these sets for overlap metrics such as Jaccard, precision, recall, and F1. These metrics are not meaningful for datasets without planted ground truth.


The original dataset files are not changed. The generated `_prepared` folder records the mappings used for each dataset.

## Outputs and metrics

With the default `root_name = "datasets"`, the runner generates:

```text
synthetic_results/
└── datasets/
    ├── _prepared/
    │   └── <dataset_folder>/
    │       ├── graph.edgelist
    │       ├── attributes.csv
    │       ├── node_mapping.csv
    │       ├── group_mapping.csv
    │       └── planted_subgraphs.csv   # only if provided
    ├── greedy/
    │   └── <dataset_folder>/
    │       └── densest_subgraph_run00.csv
    └── <other_method>/
        └── <dataset_folder>/
            └── densest_subgraph_run00.csv

datasets_results_S1.csv
```

Each extracted-subgraph CSV contains `node_id` and `attribute`. The summary CSV contains per-method averages of successfully completed runs, including:

- `density`, `diverse_density`, and `subgraph_size`.
- `group_density_min`, `group_density_max`, and `group_density_avg`.
- `group_diverse_density_min`, `group_diverse_density_max`, and `group_diverse_density_avg`.
- `runtime_sec` and run-success information.
- Ground-truth overlap metrics, when a planted set is available.

The **minimum group density** corresponds to eFairD. The **minimum group-diverse density** corresponds to eFairDivD. The main runner currently reports means of successful runs, but does not generate all the runtime, standard-deviation, exact-optimization, or comparison tables shown later in this README.

## Reproducibility and data notes

The repository provides **12 executable methods**, including density and diverse-density baselines, and two example datasets. The tables below report a wider evaluation from the manuscript and additional experiments, including methods and datasets not distributed in this repository.

Specifically, the current repository does **not** include all real datasets, all synthetic graph generators and configurations, external baseline implementations, Gurobi MILP scripts, or the complete ten-run outputs. Accordingly, `python main.py` is an entry point for the available methods and data, **not a one-command reproduction of every result in the paper or this README**.

The DBLP attribute file associates ethnicity-category labels with individual researchers. It does not contain names or DBLP profile identifiers, but coauthorship graph structure may still allow records to be linked to publicly identifiable people. The released file should not be described as proven anonymous. The dataset's label-generation methodology and whether these categories are self-identified are **not documented here**, so the labels should not be interpreted as verified personal ethnicity. Data provenance and release permissions should be reviewed separately.

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

#### Five-group homophily setting

The latest paper also reports a separate homophily experiment with **five equally sized groups** and three disjoint planted sets S₁, S₂, and S₃. Their sizes are 200, 100, and 50 nodes, with respective added-edge budgets 3,300, 700, and 300. The same-group edge fraction pₕ varies from 0.0 to 1.0 in steps of 0.1. The planted sets differ in homophily and group composition while the construction preserves the intended graph-wide homophily. Results are averaged over 10 random graphs per setting.

This five-group experiment is distinct from the two-group eFairD and three-group eFairDivD background-degree experiments above. Its complete generation scripts and sweep inputs are **not included in the present GitHub repository**.

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



