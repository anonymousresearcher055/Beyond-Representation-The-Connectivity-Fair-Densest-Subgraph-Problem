
import numpy as np
import pandas as pd
import scipy.sparse as sp
import tensorflow.compat.v2 as tf


tf.compat.v1.enable_v2_behavior()


# -----------------------------------------------------------------------------
# Graph loading
# -----------------------------------------------------------------------------

def load_graphGroup(edgelist_file, features_file, featuresType, groupAttr):

    edges = pd.read_csv(edgelist_file, sep=" ", header=None, names=["u", "v"])
    edges["u"] = pd.to_numeric(edges["u"], errors="coerce")
    edges["v"] = pd.to_numeric(edges["v"], errors="coerce")
    edges = edges.dropna(subset=["u", "v"])
    edges["u"] = edges["u"].astype(int)
    edges["v"] = edges["v"].astype(int)

    features_data = pd.read_csv(features_file)

    if "node" not in features_data.columns:
        for c in ["node_id", "node_idx"]:
            if c in features_data.columns:
                features_data = features_data.rename(columns={c: "node"})
                break
    if "node" not in features_data.columns or "attribute" not in features_data.columns:
        raise ValueError(
            f"Features file must have columns [node, attribute]. Got {features_data.columns.tolist()}"
        )

    features_data["node"] = pd.to_numeric(features_data["node"], errors="coerce")
    features_data["attribute"] = pd.to_numeric(features_data["attribute"], errors="coerce")
    features_data = features_data.dropna(subset=["node", "attribute"])
    features_data["node"] = features_data["node"].astype(int)
    features_data["attribute"] = features_data["attribute"].astype(int)

    nodes = np.unique(
        np.concatenate([edges["u"].values, edges["v"].values, features_data["node"].values])
    )
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)

    row = edges["u"].map(node_to_idx).to_numpy()
    col = edges["v"].map(node_to_idx).to_numpy()

    adjacency = sp.coo_matrix(
        (np.ones(len(edges), dtype=np.float32), (row, col)), shape=(n, n)
    ).tocsr()
    adjacency = (adjacency + adjacency.T).minimum(1)

    features_data["node_idx"] = features_data["node"].map(node_to_idx).astype(int)

    node_attribute_dict = dict(
        zip(features_data["node_idx"].values, features_data["attribute"].values)
    )
    node_attribute_df = pd.DataFrame(
        {
            "node_idx": features_data["node_idx"],
            "attribute": features_data["attribute"],
        }
    )

    coo = adjacency.tocoo()
    attr = np.full(n, -1, dtype=int)
    for i, v in node_attribute_dict.items():
        attr[int(i)] = int(v)

    groupAttr = int(groupAttr)
    mask_g = (attr[coo.row] == groupAttr) | (attr[coo.col] == groupAttr)

    group_adjacency = sp.coo_matrix(
        (coo.data[mask_g], (coo.row[mask_g], coo.col[mask_g])), shape=(n, n)
    ).tocsr()


    if featuresType == "degree":
        deg = np.asarray(adjacency.sum(axis=1)).ravel().astype(np.float32)
        features = sp.diags(deg, format="csr")
    elif featuresType == "id":
        features = sp.identity(n, format="csr", dtype=np.float32)
    else:
        x = np.zeros(n, dtype=np.float32)
        x[features_data["node_idx"].to_numpy(dtype=int)] = features_data[
            "attribute"
        ].to_numpy(dtype=np.float32)
        features = sp.csr_matrix(x.reshape(-1, 1))

    return (
        adjacency,
        group_adjacency,
        features,
        node_attribute_dict,
        node_attribute_df,
    )


def load_graphDiversity(edgelist_file, features_file, featuresType):
    edges = pd.read_csv(edgelist_file, sep=" ", header=None, names=["u", "v"])
    edges["u"] = pd.to_numeric(edges["u"], errors="coerce")
    edges["v"] = pd.to_numeric(edges["v"], errors="coerce")
    edges = edges.dropna(subset=["u", "v"])
    edges["u"] = edges["u"].astype(int)
    edges["v"] = edges["v"].astype(int)

    features_data = pd.read_csv(features_file)

    if "node" not in features_data.columns:
        for c in ["node_id", "node_idx"]:
            if c in features_data.columns:
                features_data = features_data.rename(columns={c: "node"})
                break
    if "node" not in features_data.columns or "attribute" not in features_data.columns:
        raise ValueError(
            f"Features file must have columns [node, attribute]. Got {features_data.columns.tolist()}"
        )

    features_data["node"] = pd.to_numeric(features_data["node"], errors="coerce")
    features_data = features_data.dropna(subset=["node"]).copy()
    features_data["node"] = features_data["node"].astype(int)

    attr_num = pd.to_numeric(features_data["attribute"], errors="coerce")
    if attr_num.notna().all():
        features_data["attribute"] = attr_num.astype(int)
    else:
        features_data["attribute"] = pd.Categorical(
            features_data["attribute"].astype(str)
        ).codes.astype(int)

    nodes = np.unique(
        np.concatenate([edges["u"].values, edges["v"].values, features_data["node"].values])
    )
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)

    row = edges["u"].map(node_to_idx).to_numpy()
    col = edges["v"].map(node_to_idx).to_numpy()

    adjacency = sp.coo_matrix(
        (np.ones(len(edges), dtype=np.float32), (row, col)), shape=(n, n)
    ).tocsr()
    adjacency = (adjacency + adjacency.T).minimum(1)

    features_data["node_idx"] = features_data["node"].map(node_to_idx).astype(int)

    node_attribute_dict = dict(
        zip(features_data["node_idx"].values, features_data["attribute"].values)
    )
    node_attribute_df = pd.DataFrame(
        {
            "node_idx": features_data["node_idx"],
            "attribute": features_data["attribute"],
        }
    )

    coo = adjacency.tocoo()
    attr = np.full(n, -1, dtype=int)
    for i, v in node_attribute_dict.items():
        attr[int(i)] = int(v)

    mask_div = (
        (attr[coo.row] != -1)
        & (attr[coo.col] != -1)
        & (attr[coo.row] != attr[coo.col])
    )

    diversity_adjacency = sp.coo_matrix(
        (coo.data[mask_div], (coo.row[mask_div], coo.col[mask_div])), shape=(n, n)
    ).tocsr()


    if featuresType == "degree":
        deg = np.asarray(adjacency.sum(axis=1)).ravel().astype(np.float32)
        features = sp.diags(deg, format="csr")
    elif featuresType == "id":
        features = sp.identity(n, format="csr", dtype=np.float32)
    else:
        x = np.zeros(n, dtype=np.float32)
        x[features_data["node_idx"].to_numpy(dtype=int)] = features_data[
            "attribute"
        ].to_numpy(dtype=np.float32)
        features = sp.csr_matrix(x.reshape(-1, 1))

    return (
        adjacency,
        diversity_adjacency,
        features,
        node_attribute_dict,
        node_attribute_df,
        nodes,
    )


def load_graph(edgelist_file, features_file, featuresType="id", return_nodes=True):
    edges = pd.read_csv(
        edgelist_file, sep=r"\s+", header=None, names=["u", "v"], engine="python"
    )
    edges = edges.dropna(subset=["u", "v"])
    edges["u"] = edges["u"].astype(int)
    edges["v"] = edges["v"].astype(int)

    nodes = np.unique(np.concatenate([edges["u"].values, edges["v"].values]))
    node_to_idx = {int(node): i for i, node in enumerate(nodes)}
    n = len(nodes)

    row = edges["u"].map(node_to_idx).to_numpy()
    col = edges["v"].map(node_to_idx).to_numpy()

    adjacency = sp.coo_matrix(
        (np.ones(len(edges), dtype=np.float32), (row, col)), shape=(n, n)
    ).tocsr()
    adjacency = (adjacency + adjacency.T).minimum(1)

    feat = pd.read_csv(features_file)

    node_col = None
    for c in ["node", "nodes", "node_id", "node_idx", "id"]:
        if c in feat.columns:
            node_col = c
            break
    if node_col is None or "attribute" not in feat.columns:
        raise ValueError(
            f"Attributes file must have node column + 'attribute'. Got: {feat.columns.tolist()}"
        )

    feat[node_col] = feat[node_col].astype(int)

    if feat["attribute"].dtype.kind not in "iu":
        feat["attribute"] = pd.Categorical(feat["attribute"].astype(str)).codes.astype(int)
    else:
        feat["attribute"] = feat["attribute"].astype(int)

    feat["node_idx"] = feat[node_col].map(node_to_idx)
    feat = feat.dropna(subset=["node_idx"]).copy()
    feat["node_idx"] = feat["node_idx"].astype(int)

    node_attribute_dict = dict(zip(feat["node_idx"].values, feat["attribute"].values))
    node_attribute_df = feat[["node_idx", "attribute"]].copy()


    if featuresType == "degree":
        deg = np.asarray(adjacency.sum(axis=1)).ravel().astype(np.float32)
        features = sp.diags(deg, offsets=0, format="csr")
    elif featuresType == "id":
        features = sp.diags(np.arange(n, dtype=np.float32), offsets=0, format="csr")
    else:
        attr = np.zeros(n, dtype=np.float32)
        for i, a in node_attribute_dict.items():
            attr[i] = float(a)
        features = sp.diags(attr, offsets=0, format="csr")

    return adjacency, features, node_attribute_dict, node_attribute_df, nodes




def convert_scipy_sparse_to_sparse_tensor(matrix):
    matrix = matrix.tocoo()
    st = tf.sparse.SparseTensor(
        np.vstack([matrix.row, matrix.col]).T,
        matrix.data.astype(np.float32),
        matrix.shape,
    )
    return tf.sparse.reorder(st)


# -----------------------------------------------------------------------------
# Adam optimizer
# -----------------------------------------------------------------------------

class DirectNodeLogits(tf.Module):

    def __init__(self, n_nodes, tau=0.5, init_std=0.01, seed=None, name=None):
        super().__init__(name=name)
        self.tau = tf.convert_to_tensor(tau, dtype=tf.float32)

        if seed is None:
            initial = tf.random.normal(
                [n_nodes, 2], mean=0.0, stddev=init_std, dtype=tf.float32
            )
        else:

            initial = tf.random.stateless_normal(
                [n_nodes, 2], seed=[int(seed), int(seed) + 1],
                mean=0.0, stddev=init_std, dtype=tf.float32
            )

        self.logits = tf.Variable(initial, trainable=True, name="node_logits")

    def probabilities(self):
        return tf.nn.softmax(self.logits / self.tau, axis=1)

    def membership_scores(self):
        return self.probabilities()[:, 0:1]


@tf.function
def _rayleigh(p, adjacency, eps=1e-9):
    Ap = tf.sparse.sparse_dense_matmul(adjacency, p)
    num = tf.matmul(p, Ap, transpose_a=True)
    den = tf.reduce_sum(tf.square(p)) + tf.cast(eps, p.dtype)
    return tf.squeeze(num) / den


def _direct_adam_single_objective(
    adjacency_tf,
    n_nodes,
    epochs=1000,
    lr=0.01,
    tau=0.5,
    lambda_empty=0.0,
    init_std=0.01,
    seed=None,
    log_every=100,
    clipnorm=None,
):

    scorer = DirectNodeLogits(
        n_nodes=n_nodes, tau=tau, init_std=init_std, seed=seed
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=clipnorm)

    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            p = scorer.membership_scores()
            ray = _rayleigh(p, adjacency_tf)
            loss = -ray


            if lambda_empty and lambda_empty > 0.0:
                loss = loss + tf.cast(lambda_empty, tf.float32) * (
                    -tf.math.log(tf.reduce_sum(p) + 1e-9)
                )

        grads = tape.gradient(loss, [scorer.logits])
        optimizer.apply_gradients(zip(grads, [scorer.logits]))
        return loss, ray

    for epoch in range(epochs):
        loss, ray = train_step()
        if log_every and epoch % log_every == 0:
            print(
                f"epoch {epoch}, loss: {float(loss.numpy()):.6f}, "
                f"rayleigh: {float(ray.numpy()):.6f}"
            )

    probs = scorer.probabilities().numpy()
    scores = probs[:, 0].astype(np.float64)
    return scores, probs


def _direct_adam_maxmin_objective(
    group_adjs_tf,
    n_nodes,
    epochs=1000,
    lr=0.01,
    tau=0.5,
    tau_min=1.0,
    lambda_empty=0.0,
    init_std=0.01,
    seed=None,
    log_every=100,
    clipnorm=None,
):

    scorer = DirectNodeLogits(
        n_nodes=n_nodes, tau=tau, init_std=init_std, seed=seed
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=clipnorm)
    tau_min_tf = tf.convert_to_tensor(tau_min, dtype=tf.float32)

    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            p = scorer.membership_scores()
            den = tf.reduce_sum(tf.square(p)) + 1e-9

            rays = []
            for A_g in group_adjs_tf:
                Ap = tf.sparse.sparse_dense_matmul(A_g, p)
                num = tf.matmul(p, Ap, transpose_a=True)
                rays.append(tf.squeeze(num) / den)
            rays = tf.stack(rays, axis=0)


            softmin_ray = -tau_min_tf * tf.reduce_logsumexp(-rays / tau_min_tf)
            loss = -softmin_ray

  
            if lambda_empty and lambda_empty > 0.0:
                loss = loss + tf.cast(lambda_empty, tf.float32) / (
                    tf.reduce_sum(p) + 1e-9
                )

        grads = tape.gradient(loss, [scorer.logits])
        optimizer.apply_gradients(zip(grads, [scorer.logits]))
        return loss, softmin_ray, rays

    for epoch in range(epochs):
        loss, softmin_ray, rays = train_step()
        if log_every and epoch % log_every == 0:
            print(
                f"epoch {epoch}, loss: {float(loss.numpy()):.6f}, "
                f"softmin_rayleigh: {float(softmin_ray.numpy()):.6f}, "
                f"min_rayleigh: {float(tf.reduce_min(rays).numpy()):.6f}"
            )

    probs = scorer.probabilities().numpy()
    scores = probs[:, 0].astype(np.float64)
    return scores, probs




def sweep_densest_prefix(A_csr, scores):
    n = A_csr.shape[0]
    order = np.argsort(scores)[::-1]

    in_set = np.zeros(n, dtype=bool)
    internal = 0.0

    best_density = -1.0
    best_size = 0

    indptr, indices, data = A_csr.indptr, A_csr.indices, A_csr.data

    for s, u in enumerate(order, start=1):
        start, end = indptr[u], indptr[u + 1]
        nbrs = indices[start:end]
        wts = data[start:end]
        if nbrs.size:
            internal += wts[in_set[nbrs]].sum()
        in_set[u] = True

        density = internal / s
        if density > best_density:
            best_density = density
            best_size = s

    return order[:best_size], float(best_density)


def build_group_adjacency_list_multigroup(A_csr, node_attribute_dict, incident=True):
    n = A_csr.shape[0]

    attr = np.empty(n, dtype=object)
    attr[:] = None
    for i, v in node_attribute_dict.items():
        attr[int(i)] = v

    valid = np.array([a is not None for a in attr], dtype=bool)
    groups = np.unique(attr[valid]).tolist()

    coo = A_csr.tocoo()
    row, col, data = coo.row, coo.col, coo.data

    group_adjs = []
    for g in groups:
        if incident:
            mask = (attr[row] == g) | (attr[col] == g)
        else:
            mask = (attr[row] == g) & (attr[col] == g)

        A_g = sp.coo_matrix(
            (data[mask], (row[mask], col[mask])), shape=A_csr.shape
        ).tocsr()
        group_adjs.append(A_g)

    return groups, group_adjs


def build_group_diverse_adjacency_list_multigroup(A_csr, node_attribute_dict):
    n = A_csr.shape[0]

    attr = np.empty(n, dtype=object)
    attr[:] = None
    for i, v in node_attribute_dict.items():
        attr[int(i)] = v

    valid = np.array([a is not None for a in attr], dtype=bool)
    uniq_labels, inv = np.unique(attr[valid], return_inverse=True)

    codes = np.full(n, -1, dtype=np.int32)
    codes[valid] = inv
    groups = uniq_labels.tolist()
    K = len(groups)

    coo = A_csr.tocoo()
    row, col, data = coo.row, coo.col, coo.data

    gu = codes[row]
    gv = codes[col]

    mask = (gu >= 0) & (gv >= 0) & (gu != gv)
    row = row[mask]
    col = col[mask]
    data = data[mask]
    gu = gu[mask]
    gv = gv[mask]

    grp = np.concatenate([gu, gv])
    rows2 = np.concatenate([row, row])
    cols2 = np.concatenate([col, col])
    data2 = np.concatenate([data, data])

    order = np.argsort(grp, kind="mergesort")
    grp_s = grp[order]
    rows_s = rows2[order]
    cols_s = cols2[order]
    data_s = data2[order]

    counts = np.bincount(grp_s, minlength=K)
    offsets = np.cumsum(counts)

    group_div_adjs = []
    start = 0
    for g in range(K):
        end = offsets[g]
        if end == start:
            Ag_div = sp.csr_matrix(A_csr.shape, dtype=A_csr.dtype)
        else:
            Ag_div = sp.coo_matrix(
                (data_s[start:end], (rows_s[start:end], cols_s[start:end])),
                shape=A_csr.shape,
            ).tocsr()
        group_div_adjs.append(Ag_div)
        start = end

    return groups, group_div_adjs


def _subgraph_density(A_csr, S):
    if len(S) == 0:
        return 0.0
    S = np.asarray(S, dtype=int)
    sub = A_csr[S][:, S]
    internal_edges = sub.sum() / 2.0
    return float(internal_edges / max(1, len(S)))


def sweep_maxmin_prefix_fast(group_adjs_csr, scores):
    K = len(group_adjs_csr)
    n = group_adjs_csr[0].shape[0]
    order = np.argsort(scores)[::-1]

    in_set = np.zeros(n, dtype=bool)
    internal = np.zeros(K, dtype=float)
    best_internal = None

    best_val = -1.0
    best_size = 0

    indptr_list = [Ag.indptr for Ag in group_adjs_csr]
    indices_list = [Ag.indices for Ag in group_adjs_csr]
    data_list = [Ag.data for Ag in group_adjs_csr]

    for s, u in enumerate(order, start=1):
        for j in range(K):
            indptr = indptr_list[j]
            indices = indices_list[j]
            data = data_list[j]

            start, end = indptr[u], indptr[u + 1]
            nbrs = indices[start:end]
            if nbrs.size:
                internal[j] += data[start:end][in_set[nbrs]].sum()

        in_set[u] = True

        dens = internal / float(s)
        val = float(dens.min())

        if val > best_val:
            best_val = val
            best_size = s
            best_internal = internal.copy()

    if best_size == 0:
        best_size = 1
        best_val = 0.0
        best_internal = np.zeros(K, dtype=float)

    best_nodes = order[:best_size].tolist()
    best_group_dens = best_internal / float(best_size)
    return best_nodes, best_val, best_group_dens



def AdamDensestSubgraph(
    edgelist_path,
    attributes_path,
    featuresType="id",
    epochs=1000,
    lr=0.01,
    use_abs=False,
    tau=0.5,
    lambda_empty=0.0,
    init_std=0.01,
    seed=None,
    log_every=100,
    clipnorm=None,
):

    adjacency, _, _, _, _ = load_graph(
        edgelist_path, attributes_path, featuresType
    )
    A = adjacency.tocsr()
    A_tf = convert_scipy_sparse_to_sparse_tensor(A)

    scores, probs = _direct_adam_single_objective(
        adjacency_tf=A_tf,
        n_nodes=A.shape[0],
        epochs=epochs,
        lr=lr,
        tau=tau,
        lambda_empty=lambda_empty,
        init_std=init_std,
        seed=seed,
        log_every=log_every,
        clipnorm=clipnorm,
    )

    if use_abs:
        scores = np.abs(scores)

    best_idx, best_density = sweep_densest_prefix(A, scores)
    best_nodes = best_idx.tolist()

    print("best_density:", best_density)
    return best_density, best_nodes, scores


def AdamDensestDiverseSubgraph(
    edgelist_path,
    attributes_path,
    featuresType="id",
    epochs=1000,
    lr=0.01,
    use_abs=False,
    tau=0.5,
    lambda_empty=0.0,
    init_std=0.01,
    seed=None,
    log_every=100,
    clipnorm=None,
):

    _, diversity_adjacency, _, _, _, _ = load_graphDiversity(
        edgelist_path, attributes_path, featuresType
    )
    A_div = diversity_adjacency.tocsr()
    A_div_tf = convert_scipy_sparse_to_sparse_tensor(A_div)

    scores, probs = _direct_adam_single_objective(
        adjacency_tf=A_div_tf,
        n_nodes=A_div.shape[0],
        epochs=epochs,
        lr=lr,
        tau=tau,
        lambda_empty=lambda_empty,
        init_std=init_std,
        seed=seed,
        log_every=log_every,
        clipnorm=clipnorm,
    )

    if use_abs:
        scores = np.abs(scores)

    best_idx, best_density = sweep_densest_prefix(A_div, scores)
    best_nodes = best_idx.tolist()

    print("best_density:", best_density)
    print("len(best_nodes):", len(best_nodes))
    return best_density, best_nodes, scores


def AdamDensestGroupSubgraphMaxMin(
    edgelist_path,
    attributes_path,
    featuresType="id",
    incident=True,
    epochs=1000,
    lr=0.01,
    lambda_empty=0.0,
    tau=0.5,
    tau_min=1.0,
    init_std=0.01,
    seed=None,
    log_every=100,
    clipnorm=None,
):

    adjacency, _, node_attr_dict, _, _ = load_graph(
        edgelist_path, attributes_path, featuresType
    )
    A_all = adjacency.tocsr()

    groups, group_adjs = build_group_adjacency_list_multigroup(
        A_all, node_attr_dict, incident=incident
    )
    if len(group_adjs) < 2:
        raise ValueError("Need at least 2 groups for min-group objective.")

    group_tf = [convert_scipy_sparse_to_sparse_tensor(Ag) for Ag in group_adjs]

    scores, probs = _direct_adam_maxmin_objective(
        group_adjs_tf=group_tf,
        n_nodes=A_all.shape[0],
        epochs=epochs,
        lr=lr,
        tau=tau,
        tau_min=tau_min,
        lambda_empty=lambda_empty,
        init_std=init_std,
        seed=seed,
        log_every=log_every,
        clipnorm=clipnorm,
    )

    best_nodes, best_min_group_density, best_group_densities = \
        sweep_maxmin_prefix_fast(group_adjs, scores)
    best_overall_density = _subgraph_density(A_all, best_nodes)

    print("best_eFairD:", best_min_group_density)
    print("best_group_densities:", best_group_densities)
    print("best_overall_density:", best_overall_density)


    return best_overall_density, best_nodes, scores, probs, groups


def AdamDensestGroupDiverseSubgraphMaxMin(
    edgelist_path,
    attributes_path,
    featuresType="id",
    incident=True, 
    epochs=1000,
    lr=0.01,
    lambda_empty=0.0,
    tau=0.5,
    tau_min=1.0,
    init_std=0.01,
    seed=None,
    log_every=100,
    clipnorm=None,
):

    adjacency, _, node_attr_dict, _, _ = load_graph(
        edgelist_path, attributes_path, featuresType
    )
    A_all = adjacency.tocsr()

    groups, group_adjs = build_group_diverse_adjacency_list_multigroup(
        A_all, node_attr_dict
    )
    if len(group_adjs) < 2:
        raise ValueError("Need at least 2 groups for min-group objective.")

    group_tf = [convert_scipy_sparse_to_sparse_tensor(Ag) for Ag in group_adjs]

    scores, probs = _direct_adam_maxmin_objective(
        group_adjs_tf=group_tf,
        n_nodes=A_all.shape[0],
        epochs=epochs,
        lr=lr,
        tau=tau,
        tau_min=tau_min,
        lambda_empty=lambda_empty,
        init_std=init_std,
        seed=seed,
        log_every=log_every,
        clipnorm=clipnorm,
    )

    best_nodes, best_min_group_density, best_group_densities = \
        sweep_maxmin_prefix_fast(group_adjs, scores)
    best_overall_density = _subgraph_density(A_all, best_nodes)

    print("best_eFairDivD:", best_min_group_density)
    print("best_group_diverse_densities:", best_group_densities)
    print("best_overall_density:", best_overall_density)


    return best_overall_density, best_nodes, scores, probs, groups
