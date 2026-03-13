
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
import networkx as nx

from spectral_densest_subgraph import spectral_densest_subgraph_from_adjacency


import sys
sys.path.append("..")  

from greedy_algortihm import (  
    _greedy_plus_plus,
    diverse_greedy_plus_plus,
    compute_density,
    compute_diverse_density,
    maxmin_greedy_plus_plus,
    compute_group_density,
    compute_group_diverse_density,
    maxmin_group_diverse_greedy_plus_plus,
)

sys.path.append("Deep Dense subgraph")
from deepDenseSubgraph import (
    deepDensestSubgraph,
    deepDensestGroupSubgraphMaxMin,
    deepDensestGroupDiverseSubgraphMaxMin,
    deepDensestDiverseSubgraph,

)


STest = "S1"

root_name = "datasets"


DATASETS_ROOT = Path(f"{root_name}")

N_RUNS_PER_METHOD = 1
OUT_ROOT = Path("synthetic_results") / root_name





def base_dataset_name(folder_name: str) -> str:
    s = folder_name
    s = re.sub(r"__seed\d+", "", s)
    s = re.sub(r"__run\d+", "", s)
    s = re.sub(r"__+", "__", s).strip("_")
    return s



def find_dataset_files(ds_dir: Path) -> Tuple[Path, Path, Optional[Path]]:
    ds_name = ds_dir.name


    cand_edges = [
        ds_dir / "graph.edgelist",
        ds_dir / f"{ds_name}.edgelist",
        ds_dir / "edgelist.txt",
    ]
    edgelist_path = None
    for p in cand_edges:
        if p.exists():
            edgelist_path = p
            break
    if edgelist_path is None:
        gl = sorted(ds_dir.glob("*.edgelist"))
        if gl:
            edgelist_path = gl[0]
    if edgelist_path is None:
        raise FileNotFoundError(f"No edgelist file found in {ds_dir}")


    cand_attrs = [
        ds_dir / "attributes.csv",
        ds_dir / f"{ds_name}_attributes.csv",
    ]
    attributes_path = None
    for p in cand_attrs:
        if p.exists():
            attributes_path = p
            break
    if attributes_path is None:
        gl = sorted([
            p for p in ds_dir.glob("*.csv")
            if p.name not in {
                "planted_subgraphs.csv",
                "planted_inputs.csv",
                "graph_metrics.csv",
                "planted_metrics.csv",
            }
            and "metrics" not in p.name.lower()
        ])
        if gl:
            attributes_path = gl[0]
    if attributes_path is None:
        raise FileNotFoundError(f"No attributes csv found in {ds_dir}")

    cand_gt = [
        ds_dir / "planted_subgraphs.csv",
    ]
    gt_path = None
    for p in cand_gt:
        if p.exists():
            gt_path = p
            break
    if gt_path is None:
        gl = sorted([p for p in ds_dir.glob("*.csv") if "subgraph" in p.name.lower()])
        if gl:
            gt_path = gl[0]

    return edgelist_path, attributes_path, gt_path


def load_graph_from_synthetic_files(
    edgelist_file: Path,
    attributes_file: Path,
    gt_file: Optional[Path] = None,
):
    feat = pd.read_csv(attributes_file)

    node_col = None
    for c in ["node_id", "node", "node_idx"]:
        if c in feat.columns:
            node_col = c
            break

    if node_col is None or "attribute" not in feat.columns:
        raise ValueError(
            f"Attributes file must have columns [node_id/node/node_idx, attribute]. "
            f"Got: {feat.columns.tolist()}"
        )

    feat[node_col] = feat[node_col].astype(int)
    feat["attribute"] = feat["attribute"].astype(int)

    n = int(feat[node_col].max()) + 1
    node_attributes_dict = dict(zip(feat[node_col].values, feat["attribute"].values))

    edges = pd.read_csv(edgelist_file, sep=r"\s+", header=None, names=["u", "v"])
    edges["u"] = edges["u"].astype(int)
    edges["v"] = edges["v"].astype(int)

    row = edges["u"].values
    col = edges["v"].values
    data = np.ones(len(edges), dtype=np.float32)

    adjacency = sp.coo_matrix((data, (row, col)), shape=(n, n)).tocsr()
    adjacency = (adjacency + adjacency.T).minimum(1)
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()

    G = nx.Graph()
    G.add_nodes_from(range(n))
    coo = adjacency.tocoo()
    for u, v in zip(coo.row, coo.col):
        if u < v:
            G.add_edge(int(u), int(v))

    nx.set_node_attributes(G, node_attributes_dict, "attribute")

    densest_sets: Dict[str, List[int]] = {}
    if gt_file is not None and Path(gt_file).exists():
        gt_df = pd.read_csv(gt_file)

        set_col = None
        for c in ["set_name", "set", "subgraph", "subgraph_id", "planted_set", "community"]:
            if c in gt_df.columns:
                set_col = c
                break

        node_col_gt = None
        for c in ["node_id", "node", "vertex", "v", "u", "node_idx"]:
            if c in gt_df.columns:
                node_col_gt = c
                break

        if node_col_gt is not None:
            gt_df[node_col_gt] = gt_df[node_col_gt].astype(int)
            if set_col is not None:
                densest_sets = {
                    str(name): grp[node_col_gt].drop_duplicates().tolist()
                    for name, grp in gt_df.groupby(set_col)
                }
            else:
                densest_sets = {"GT": gt_df[node_col_gt].drop_duplicates().tolist()}

    return G, adjacency, node_attributes_dict, densest_sets



def overlap_stats(pred_nodes, gt_nodes):
    P = set(pred_nodes)
    Gt = set(gt_nodes)

    inter = len(P & Gt)
    union = len(P | Gt)

    jaccard = inter / union if union else 0.0
    precision = inter / len(P) if len(P) else 0.0
    recall = inter / len(Gt) if len(Gt) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "gt_intersection": inter,
        "gt_union": union,
        "gt_jaccard": jaccard,
        "gt_precision": precision,
        "gt_recall": recall,
        "gt_f1": f1,
    }


def choose_gt_target_set(
    pred_subgraphs: List[List[int]],
    groundTruth_dict: Dict[str, List[int]],
    score_key: str = "gt_jaccard",
) -> Optional[str]:
    if not groundTruth_dict or not pred_subgraphs:
        return None

    best_name = None
    best_score = -1.0

    for name, gt_nodes in groundTruth_dict.items():
        scores = [overlap_stats(pred, gt_nodes).get(score_key, 0.0) for pred in pred_subgraphs]
        mean_score = float(np.mean(scores)) if scores else 0.0
        if mean_score > best_score:
            best_score = mean_score
            best_name = name

    return best_name


def resolve_gt_target_for_method(
    method_name: str,
    groundTruth_dict: Dict[str, List[int]],
    pred_subgraphs: List[List[int]],
    gt_target_by_method: Optional[Dict[str, str]] = None,
    auto_score_key: str = "gt_jaccard",
) -> Optional[str]:
    if not groundTruth_dict:
        return None

    if gt_target_by_method and method_name in gt_target_by_method:
        chosen = gt_target_by_method[method_name]
        if chosen not in groundTruth_dict:
            raise ValueError(
                f"Method '{method_name}' requested GT set '{chosen}', "
                f"but available GT sets are: {list(groundTruth_dict.keys())}"
            )
        return chosen

    return choose_gt_target_set(pred_subgraphs, groundTruth_dict, score_key=auto_score_key)



def compute_metrics_row(G, best_subgraph, node_attributes_dict, runtime_sec=None):
    density = compute_density(G, best_subgraph)
    diverse_density = compute_diverse_density(G, best_subgraph, node_attributes_dict, lam=1)

    groups = sorted(set(node_attributes_dict[n] for n in G.nodes))

    group_densities = [
        compute_group_density(G, best_subgraph, node_attributes_dict, g)
        for g in groups
    ]
    group_diverse_densities = [
        compute_group_diverse_density(G, best_subgraph, node_attributes_dict, g)
        for g in groups
    ]

    if len(groups) == 0:
        gd_min = gd_max = gd_avg = np.nan
        gdd_min = gdd_max = gdd_avg = np.nan
    else:
        gd_min = float(np.min(group_densities))
        gd_max = float(np.max(group_densities))
        gd_avg = float(np.mean(group_densities))

        gdd_min = float(np.min(group_diverse_densities))
        gdd_max = float(np.max(group_diverse_densities))
        gdd_avg = float(np.mean(group_diverse_densities))

    row = {
        "density": density,
        "group_density_min": gd_min,
        "group_density_max": gd_max,
        "group_density_avg": gd_avg,
        "group_diverse_density_min": gdd_min,
        "group_diverse_density_max": gdd_max,
        "group_diverse_density_avg": gdd_avg,
        "diverse_density": diverse_density,
        "subgraph_size": len(best_subgraph),
        "runtime_sec": runtime_sec,
    }
    return row



def run_greedy(G):
    _, S = _greedy_plus_plus(G, 1)
    return S


def run_greedy_diverse(G, node_attributes_dict):
    _, S = diverse_greedy_plus_plus(G, node_attributes_dict, 1)
    return S


def run_maxmin_greedy_plus_plus(G, node_attributes_dict):
    _, S = maxmin_greedy_plus_plus(G, node_attributes_dict, 1)
    return S




def run_diverse_maxmin_greedy_plus_plus(G, node_attributes_dict):
    _, S = maxmin_group_diverse_greedy_plus_plus(G, node_attributes_dict, 1)
    return S


def run_spectral(adjacency):
    _, S = spectral_densest_subgraph_from_adjacency(adjacency)
    return S


def _attrs_array(node_attributes_dict, n):
    return np.array([node_attributes_dict.get(i, np.nan) for i in range(n)], dtype=object)


def build_diverse_adjacency(A_csr, node_attributes_dict):
    A = A_csr.tocoo()
    n = A.shape[0]
    attrs = _attrs_array(node_attributes_dict, n)

    ar = attrs[A.row]
    ac = attrs[A.col]

    keep = (ar != ac) & (~pd.isna(ar)) & (~pd.isna(ac))

    rows = A.row[keep]
    cols = A.col[keep]
    data = np.ones(rows.shape[0], dtype=np.float32)

    D = sp.coo_matrix((data, (rows, cols)), shape=A.shape).tocsr()
    D = D.maximum(D.T)
    D.setdiag(0)
    D.eliminate_zeros()
    return D







def run_diverse_spectral(div_adj):
    _, S = spectral_densest_subgraph_from_adjacency(div_adj)
    return S


def run_deep(file_path, dfFile_path):
    _, S, _ = deepDensestSubgraph(file_path, dfFile_path)
    return S




def run_deepDiverse(file_path, dfFile_path):
    _, S, _ = deepDensestDiverseSubgraph(file_path, dfFile_path)
    return S






def run_deepGroupMaxMin(file_path, dfFile_path):
    _, S, _, _, _ = deepDensestGroupSubgraphMaxMin(file_path, dfFile_path)
    return S



def run_deepGroupDiverseMaxMin(file_path, dfFile_path):
    _, S, _, _, _ = deepDensestGroupDiverseSubgraphMaxMin(file_path, dfFile_path)
    return S


def save_densest_subgraph_csv(
    out_root: str | Path,
    method_name: str,
    dataset_name: str,
    run_idx: int,
    subgraph_nodes,
    node_attributes_dict: dict[int, int] | None = None,
) -> Path:
    out_root = Path(out_root)
    out_dir = out_root / method_name / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if subgraph_nodes is None:
        nodes = []
    elif isinstance(subgraph_nodes, (set, list, tuple)):
        nodes = sorted(int(x) for x in subgraph_nodes)
    else:
        try:
            nodes = sorted(int(x) for x in list(subgraph_nodes))
        except Exception:
            nodes = [int(subgraph_nodes)]

    df = pd.DataFrame({"node_id": nodes})
    if node_attributes_dict is not None:
        df["attribute"] = df["node_id"].map(lambda u: node_attributes_dict.get(int(u), None))

    out_path = out_dir / f"densest_subgraph_run{run_idx:02d}.csv"
    df.to_csv(out_path, index=False)
    return out_path



def run_and_record_avg(
    dataset_folder: str,
    dataset_base: str,
    method_name: str,
    runner,
    n_runs: int,
    G,
    node_attributes_dict,
    groundTruth_dict,
    gt_target_by_method: Optional[Dict[str, str]] = None,
    auto_gt_score_key: str = "gt_jaccard",
):
    per_run_rows: List[Dict] = []
    pred_subgraphs: List[List[int]] = []
    errors = 0

    for r in range(n_runs):
        t0 = time.time()
        try:
            best_subgraph = runner()
        except Exception as e:
            errors += 1
            print(f"[WARN] {dataset_folder} | {method_name} run {r+1}/{n_runs} failed: {e}")
            continue
        t1 = time.time()

        save_densest_subgraph_csv(
            out_root=OUT_ROOT,
            method_name=method_name,
            dataset_name=dataset_folder,
            run_idx=r,
            subgraph_nodes=best_subgraph,
            node_attributes_dict=node_attributes_dict,
        )

        pred_subgraphs.append(list(best_subgraph))
        per_run_rows.append(
            compute_metrics_row(
                G,
                best_subgraph,
                node_attributes_dict,
                runtime_sec=t1 - t0,
            )
        )

    if not per_run_rows:
        return {
            "dataset_folder": dataset_folder,
            "dataset": dataset_base,
            "method": method_name,
            "n_runs": n_runs,
            "n_success": 0,
            "n_fail": errors,
            "gt_target_set": None,
        }

    gt_target = resolve_gt_target_for_method(
        method_name=method_name,
        groundTruth_dict=groundTruth_dict,
        pred_subgraphs=pred_subgraphs,
        gt_target_by_method=gt_target_by_method,
        auto_score_key=auto_gt_score_key,
    )

    if gt_target is not None:
        gt_nodes = groundTruth_dict[gt_target]
        for i, pred in enumerate(pred_subgraphs):
            per_run_rows[i].update(overlap_stats(pred, gt_nodes))
        for i in range(len(per_run_rows)):
            per_run_rows[i]["gt_target_set"] = gt_target
    else:
        for i in range(len(per_run_rows)):
            per_run_rows[i]["gt_target_set"] = None

    df = pd.DataFrame(per_run_rows)
    avg_row = df.mean(numeric_only=True).to_dict()
    avg_row["gt_target_set"] = gt_target

    avg_row.update({
        "dataset_folder": dataset_folder,
        "dataset": dataset_base,
        "method": method_name,
        "n_runs": n_runs,
        "n_success": len(per_run_rows),
        "n_fail": errors,
    })
    return avg_row




GT_TARGET_BY_METHOD = {
    "deep": STest,
    "greedy": STest,
    "spectral": STest,
    "deepGroupMaxMin": STest,
    "maxmin_greedy": STest,
    "greedy_diverse": STest,
    "maxmin_diverse_greedy": STest,
    "spectral_diverse": STest,
    "deepDiverse": STest,
    "deepGroupDiverseMaxMin": STest,
}



def main():
    ds_dirs = sorted([p for p in DATASETS_ROOT.iterdir() if p.is_dir()])
    if not ds_dirs:
        raise FileNotFoundError(f"No dataset folders found in {DATASETS_ROOT}")

    results_rows = []
    print(root_name)

    for ds_dir in ds_dirs:
        folder_name = ds_dir.name
        if "_averages" in folder_name:
            continue

        base_name = base_dataset_name(folder_name)

        edgelist_path, attr_path, gt_path = find_dataset_files(ds_dir)

        print(f"\n[DATASET] {folder_name}")
        print(f"  base: {base_name}")
        print(f"  edgelist: {edgelist_path.name}")
        print(f"  attrs:    {attr_path.name}")
        print(f"  gt:       {gt_path.name if gt_path else None}")

        G, adjacency, node_attributes_dict, groundTruth_dict = load_graph_from_synthetic_files(
            edgelist_file=edgelist_path,
            attributes_file=attr_path,
            gt_file=gt_path,
        )



        div_adj = build_diverse_adjacency(adjacency, node_attributes_dict)





        has_gt = bool(groundTruth_dict)
        print("GT available:", has_gt, "| sets:", list(groundTruth_dict.keys()) if has_gt else [])


        methods = [
            ("greedy", lambda G=G: run_greedy(G)),
            ("greedy_diverse", lambda G=G, attrs=node_attributes_dict: run_greedy_diverse(G, attrs)),
            ("maxmin_diverse_greedy", lambda G=G, attrs=node_attributes_dict: run_diverse_maxmin_greedy_plus_plus(G, attrs)),
            ("maxmin_greedy", lambda G=G, attrs=node_attributes_dict: run_maxmin_greedy_plus_plus(G, attrs)),
            ("spectral", lambda A=adjacency: run_spectral(A)),
            ("spectral_diverse", lambda A=div_adj: run_diverse_spectral(A)),
            ("deep", lambda e=str(edgelist_path), a=str(attr_path): run_deep(e, a)),
            ("deepDiverse", lambda e=str(edgelist_path), a=str(attr_path): run_deepDiverse(e, a)),
            ("deepGroupDiverseMaxMin", lambda e=str(edgelist_path), a=str(attr_path): run_deepGroupDiverseMaxMin(e, a)),
            ("deepGroupMaxMin", lambda e=str(edgelist_path), a=str(attr_path): run_deepGroupMaxMin(e, a)),
        ]


        for method_name, fn in methods:
            print("Method:", method_name)

            row = run_and_record_avg(
                dataset_folder=folder_name,
                dataset_base=base_name,
                method_name=method_name,
                runner=fn,
                n_runs=N_RUNS_PER_METHOD,
                G=G,
                node_attributes_dict=node_attributes_dict,
                groundTruth_dict=groundTruth_dict,
                gt_target_by_method=GT_TARGET_BY_METHOD,
                auto_gt_score_key="gt_jaccard",
            )
            results_rows.append(row)

    results_df = pd.DataFrame(results_rows)


    group_cols = ["dataset", "method"]
    numeric_cols = results_df.select_dtypes(include="number").columns.tolist()
    metric_numeric_cols = [c for c in numeric_cols if c not in ["n_runs", "n_success", "n_fail"]]

    if "gt_target_set" in results_df.columns:
        mode_df = (
            results_df.groupby(group_cols)["gt_target_set"]
            .agg(lambda x: x.dropna().mode().iloc[0] if x.dropna().size and x.dropna().mode().size else None)
            .reset_index()
            .rename(columns={"gt_target_set": "gt_target_set_mode"})
        )

        tmp = results_df.merge(mode_df, on=group_cols, how="left")
        keep = tmp["gt_target_set_mode"].isna() | (tmp["gt_target_set"] == tmp["gt_target_set_mode"])
        filt = tmp[keep].copy()

        avg_df = filt.groupby(group_cols, as_index=False)[metric_numeric_cols].mean()

        rep_total = results_df.groupby(group_cols).size().reset_index(name="n_replicates_total")
        rep_used = filt.groupby(group_cols).size().reset_index(name="n_replicates_used")

        avg_df = avg_df.merge(rep_total, on=group_cols, how="left")
        avg_df = avg_df.merge(rep_used, on=group_cols, how="left")
        avg_df = avg_df.merge(mode_df, on=group_cols, how="left")
        avg_df = avg_df.rename(columns={"gt_target_set_mode": "gt_target_set"})
    else:
        avg_df = results_df.groupby(group_cols, as_index=False)[metric_numeric_cols].mean()
        rep_counts = results_df.groupby(group_cols).size().reset_index(name="n_replicates")
        avg_df = avg_df.merge(rep_counts, on=group_cols, how="left")

    out2 = Path(f"{root_name}_results_{STest}.csv")
    avg_df.to_csv(out2, index=False)
    print("Saved:", out2)


if __name__ == "__main__":
    main()