import torch
import numpy as np
import os
from utils.reranking import re_ranking


def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(1, -2, qf, gf.t())
    return dist_mat.cpu().numpy()

def cosine_similarity(qf, gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    # distmat g
    #    q    1 3 2 4
    #         4 1 2 3
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    #  0 2 1 3
    #  1 2 3 0
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]  # select one row
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        #tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
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

def eval_func_rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True):
    """
    Rank evaluation directly from similarity matrix.
    Returns metrics in percentage scale: 0..100
    """
    if not torch.is_tensor(similarity):
        similarity = torch.tensor(similarity)

    similarity = similarity.detach().cpu()

    if not torch.is_tensor(q_pids):
        q_pids = torch.tensor(q_pids, dtype=torch.long)
    else:
        q_pids = q_pids.detach().cpu().long()

    if not torch.is_tensor(g_pids):
        g_pids = torch.tensor(g_pids, dtype=torch.long)
    else:
        g_pids = g_pids.detach().cpu().long()

    max_rank = min(max_rank, similarity.size(1))

    if get_mAP:
        indices = torch.argsort(similarity, dim=1, descending=True)
    else:
        _, indices = torch.topk(
            similarity, k=max_rank, dim=1, largest=True, sorted=True
        )

    pred_labels = g_pids[indices]
    matches = pred_labels.eq(q_pids.view(-1, 1))

    all_cmc = matches[:, :max_rank].cumsum(1)
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100.0

    if not get_mAP:
        return all_cmc, indices

    num_rel = matches.sum(1)
    valid = num_rel > 0
    assert valid.any(), "Error: all query identities do not appear in gallery"

    matches = matches[valid]
    num_rel = num_rel[valid]
    indices = indices[valid]

    tmp_cmc = matches.cumsum(1)

    inp = []
    for i, match_row in enumerate(matches):
        pos_idx = torch.nonzero(match_row, as_tuple=False).squeeze(1)
        last_pos = pos_idx[-1]
        inp.append(tmp_cmc[i, last_pos].float() / (last_pos.float() + 1.0))
    mINP = torch.stack(inp).mean() * 100.0

    precision = tmp_cmc.float() / torch.arange(
        1, tmp_cmc.shape[1] + 1, dtype=torch.float32
    ).unsqueeze(0)
    AP = (precision * matches.float()).sum(1) / num_rel.float()
    mAP = AP.mean() * 100.0

    return all_cmc, mAP, mINP, indices


def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True):
    return eval_func_rank(
        similarity=similarity,
        q_pids=q_pids,
        g_pids=g_pids,
        max_rank=max_rank,
        get_mAP=get_mAP
    )

class R1_mAP_eval():
    def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=False):
        super(R1_mAP_eval, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []
    
    def reset_gallery(self):
        self.feats_gallery = []
        self.pids_gallery = []
        self.camids_gallery = []
        
    def update(self, output):  # called once for each batch
        feat, pid, camid = output
        self.feats.append(feat.cpu())
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def update_gallery(self, output):  # called once for each batch
        feat, pid, camid = output
        self.feats_gallery.append(feat.cpu())
        self.pids_gallery.extend(np.asarray(pid))
        self.camids_gallery.extend(np.asarray(camid))

    def compute(self):  # called after each epoch
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        if self.reranking:
            print('=> Enter reranking')
            # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)

        else:
            print('=> Computing DistMat with euclidean_distance')
            distmat = euclidean_distance(qf, gf)
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP, distmat, self.pids, self.camids, qf, gf

    def compute_cross_cls(self):
        feats = torch.cat(self.feats, dim=0)
        feats_gallery = torch.cat(self.feats_gallery, dim=0)
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, p=2, dim=1)
            feats_gallery = torch.nn.functional.normalize(feats_gallery, p=2, dim=1)
        
        # query
        qf = feats
        q_pids = np.asarray(self.pids)
        q_camids = np.asarray(self.camids)
        # gallery
        gf = feats_gallery
        g_pids = np.asarray(self.pids_gallery)
        g_camids = np.asarray(self.camids_gallery)
        
        sims = qf @ gf.t()
        
        cmc, mAP, mINP, indices = eval_func_rank(
            similarity=sims,
            q_pids=q_pids,
            g_pids=g_pids,
            max_rank=self.max_rank,
            get_mAP=True
        )
        return cmc, mAP, mINP, sims, q_pids, g_pids, q_camids, g_camids, qf, gf, indices