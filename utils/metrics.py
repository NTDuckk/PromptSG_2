import torch
import numpy as np
from utils.reranking import re_ranking


def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(qf, gf.t(), beta=1, alpha=-2)
    return dist_mat.cpu().numpy()


def cosine_distance(qf, gf):
    qf = torch.nn.functional.normalize(qf, dim=1)
    gf = torch.nn.functional.normalize(gf, dim=1)
    sim = qf.mm(gf.t())
    dist = 1.0 - sim
    return dist.cpu().numpy()


def cosine_similarity(qf, gf):
    qf = torch.nn.functional.normalize(qf, dim=1)
    gf = torch.nn.functional.normalize(gf, dim=1)
    return qf.mm(gf.t()).cpu().numpy()


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric.
    For each query identity, gallery images from the same camera are discarded.
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))

    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_cmc = []
    all_AP = []
    num_valid_q = 0.0
    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.0

        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    return all_cmc, mAP


def compute_reid_metrics(
    qf,
    gf,
    q_pids,
    g_pids,
    q_camids,
    g_camids,
    max_rank=50,
    metric="cosine",
    normalize=True,
    reranking=False,
):
    if not isinstance(qf, torch.Tensor):
        qf = torch.tensor(qf)
    if not isinstance(gf, torch.Tensor):
        gf = torch.tensor(gf)

    qf = qf.float().cpu()
    gf = gf.float().cpu()
    if normalize:
        qf = torch.nn.functional.normalize(qf, dim=1, p=2)
        gf = torch.nn.functional.normalize(gf, dim=1, p=2)

    if reranking:
        if metric != "euclidean":
            raise ValueError("reranking currently supports euclidean metric only")
        distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)
    else:
        if metric == "euclidean":
            distmat = euclidean_distance(qf, gf)
        elif metric == "cosine":
            distmat = cosine_distance(qf, gf)
        else:
            raise ValueError(f"Unsupported metric: {metric}")

    cmc, mAP = eval_func(
        distmat,
        np.asarray(q_pids),
        np.asarray(g_pids),
        np.asarray(q_camids),
        np.asarray(g_camids),
        max_rank=max_rank,
    )

    return {
        "cmc": cmc,
        "mAP": mAP,
        "distmat": distmat,
        "qf": qf,
        "gf": gf,
        "metric": metric,
    }


def format_reid_result(name, result):
    cmc = result["cmc"]
    return (
        f"{name} | mAP: {result['mAP']:.1%} | "
        f"Rank-1: {cmc[0]:.1%} | Rank-5: {cmc[4]:.1%} | Rank-10: {cmc[9]:.1%}"
    )


class R1_mAP_eval:
    """Backward-compatible evaluator for single feature stream."""
    def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=False, metric="euclidean"):
        super().__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking
        self.metric = metric

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat.cpu())
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        qf = feats[:self.num_query]
        gf = feats[self.num_query:]
        q_pids = np.asarray(self.pids[:self.num_query])
        g_pids = np.asarray(self.pids[self.num_query:])
        q_camids = np.asarray(self.camids[:self.num_query])
        g_camids = np.asarray(self.camids[self.num_query:])

        result = compute_reid_metrics(
            qf,
            gf,
            q_pids,
            g_pids,
            q_camids,
            g_camids,
            max_rank=self.max_rank,
            metric=self.metric,
            normalize=self.feat_norm,
            reranking=self.reranking,
        )
        return result["cmc"], result["mAP"], result["distmat"], self.pids, self.camids, result["qf"], result["gf"]
