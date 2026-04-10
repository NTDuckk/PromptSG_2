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

def eval_func_rank(similarity, q_pids, g_pids, q_camids, g_camids, max_rank=50, get_mAP=True):
    """
    Evaluate with camera view filtering, using similarity matrix (larger = more similar).
    Returns:
        cmc (0..100), mAP (0..100), mINP (0..100), indices (list of tensors for valid queries)
    """
    if not torch.is_tensor(similarity):
        similarity = torch.tensor(similarity)
    similarity = similarity.detach().cpu()
    q_pids = torch.tensor(q_pids, dtype=torch.long) if not torch.is_tensor(q_pids) else q_pids.detach().cpu().long()
    g_pids = torch.tensor(g_pids, dtype=torch.long) if not torch.is_tensor(g_pids) else g_pids.detach().cpu().long()
    q_camids = torch.tensor(q_camids, dtype=torch.long) if not torch.is_tensor(q_camids) else q_camids.detach().cpu().long()
    g_camids = torch.tensor(g_camids, dtype=torch.long) if not torch.is_tensor(g_camids) else g_camids.detach().cpu().long()

    num_q, num_g = similarity.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))

    # Sort descending because larger similarity = better
    indices = torch.argsort(similarity, dim=1, descending=True)  # (num_q, num_g)

    all_cmc = []
    all_AP = []
    all_matches = []
    all_valid_indices = []
    num_valid_q = 0

    for q_idx in range(num_q):
        q_pid = q_pids[q_idx].item()
        q_camid = q_camids[q_idx].item()
        order = indices[q_idx]  # (num_g,)

        # Remove gallery samples with same pid and same camid
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = ~remove
        order_keep = order[keep]

        # Binary matches after filtering
        matches = (g_pids[order_keep] == q_pid).byte()
        if matches.sum() == 0:
            continue

        num_valid_q += 1

        # CMC
        cmc = matches.cumsum(dim=0)
        cmc = (cmc > 0).float()
        all_cmc.append(cmc[:max_rank])

        # AP
        num_rel = matches.sum().float()
        tmp_cmc = matches.cumsum(dim=0).float()
        y = torch.arange(1, len(tmp_cmc) + 1, dtype=torch.float32)
        precision = tmp_cmc / y
        AP = (precision * matches.float()).sum() / num_rel
        all_AP.append(AP.item())

        all_matches.append(matches)
        all_valid_indices.append(order_keep)

    assert num_valid_q > 0, "No valid query"

    cmc = torch.stack(all_cmc).float().mean(dim=0) * 100.0   # to percentage
    mAP = np.mean(all_AP) * 100.0

    if not get_mAP:
        return cmc, None, None, None

    # mINP: mean Inverse Negative Penalty
    inp_list = []
    for matches in all_matches:
        pos_idx = torch.nonzero(matches, as_tuple=False).squeeze(1)
        if pos_idx.numel() == 0:
            continue
        last_pos = pos_idx[-1].item()
        tmp_cmc = matches.cumsum(dim=0).float()
        inp = tmp_cmc[last_pos] / (last_pos + 1.0)
        inp_list.append(inp)
    mINP = torch.stack(inp_list).mean().item() * 100.0 if inp_list else 0.0

    return cmc, mAP, mINP, all_valid_indices

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

    def compute1(self):  # called after each epoch - using cosine distance
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
            # re_ranking vẫn dùng feature gốc (không phải distance) nên giữ nguyên
            distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)
        else:
            print('=> Computing DistMat with cosine distance (1 - similarity)')
            # Vì qf và gf đã được chuẩn hóa L2, cosine similarity = qf @ gf.t()
            sim = qf.mm(gf.t())  # shape (num_query, num_gallery)
            # Chuyển similarity thành khoảng cách (càng nhỏ càng giống)
            distmat = 1 - sim
            distmat = distmat.cpu().numpy()
        
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
        return cmc, mAP, distmat, self.pids, self.camids, qf, gf

    def compute_cross_cls(self):
        feats = torch.cat(self.feats, dim=0)
        feats_gallery = torch.cat(self.feats_gallery, dim=0)
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, p=2, dim=1)
            feats_gallery = torch.nn.functional.normalize(feats_gallery, p=2, dim=1)

        qf = feats
        q_pids = np.asarray(self.pids)
        q_camids = np.asarray(self.camids)
        gf = feats_gallery
        g_pids = np.asarray(self.pids_gallery)
        g_camids = np.asarray(self.camids_gallery)

        sims = qf @ gf.t()   # cosine similarity (already normalized)

        cmc, mAP, mINP, indices = eval_func_rank(
            similarity=sims,
            q_pids=q_pids,
            g_pids=g_pids,
            q_camids=q_camids,
            g_camids=g_camids,
            max_rank=self.max_rank,
            get_mAP=True
        )
        return cmc, mAP, mINP, sims, q_pids, g_pids, q_camids, g_camids, qf, gf, indices