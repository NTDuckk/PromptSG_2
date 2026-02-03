# """
# Author: Yonglong Tian (yonglong@mit.edu)
# Date: May 07, 2020
# """
# from __future__ import print_function

# import torch
# import torch.nn as nn

# class SupConLoss(nn.Module):
#     def __init__(self, device):
#         super(SupConLoss, self).__init__()
#         self.device = device
#         self.temperature = 1.0
#     def forward(self, text_features, image_features, t_label, i_targets): 
#         batch_size = text_features.shape[0] 
#         batch_size_N = image_features.shape[0] 
#         mask = torch.eq(t_label.unsqueeze(1).expand(batch_size, batch_size_N), \
#             i_targets.unsqueeze(0).expand(batch_size,batch_size_N)).float().to(self.device) 

#         logits = torch.div(torch.matmul(text_features, image_features.T),self.temperature)
#         # for numerical stability
#         logits_max, _ = torch.max(logits, dim=1, keepdim=True)
#         logits = logits - logits_max.detach() 
#         exp_logits = torch.exp(logits) 
#         log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True)) 
#         mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1) 
#         loss = - mean_log_prob_pos.mean()

#         return loss

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class SupConLoss(nn.Module):
#     def __init__(self, device, temperature=0.07, eps=1e-12, normalize=True):
#         super().__init__()
#         self.device = device
#         self.temperature = temperature
#         self.eps = eps
#         self.normalize = normalize

#     def forward(self, anchor_feat, cand_feat, anchor_label, cand_label):
#         if self.normalize:
#             anchor_feat = F.normalize(anchor_feat, dim=1)
#             cand_feat   = F.normalize(cand_feat,   dim=1)

#         B = anchor_feat.size(0)
#         N = cand_feat.size(0)

#         mask = torch.eq(
#             anchor_label.view(B, 1).expand(B, N),
#             cand_label.view(1, N).expand(B, N)
#         ).float().to(anchor_feat.device)

#         logits = (anchor_feat @ cand_feat.t()) / self.temperature
#         logits = logits - logits.max(dim=1, keepdim=True)[0].detach()

#         log_prob = logits - torch.log(torch.exp(logits).sum(dim=1, keepdim=True) + self.eps)

#         denom = mask.sum(dim=1) + self.eps
#         mean_log_prob_pos = (mask * log_prob).sum(dim=1) / denom

#         return -mean_log_prob_pos.mean()

class SupConLoss(nn.Module):
    def __init__(self, device, temperature=0.07, eps=1e-12, normalize=True):
        super().__init__()
        self.device = device
        self.temperature = temperature
        self.eps = eps
        self.normalize = normalize

    def forward(self, anchor_feat, cand_feat, anchor_label, cand_label):
        """
        Supervised Contrastive Loss (Equation 5-6 in paper)
        
        Args:
            anchor_feat: (B1, D) - features của anchor (ví dụ: images)
            cand_feat: (B2, D)   - features của candidate (ví dụ: texts)
            anchor_label: (B1,) - labels của anchor
            cand_label: (B2,)   - labels của candidate
        """
        if self.normalize:
            anchor_feat = F.normalize(anchor_feat, dim=1)
            cand_feat = F.normalize(cand_feat, dim=1)

        B1 = anchor_feat.size(0)
        B2 = cand_feat.size(0)
        
        # Tính similarity matrix
        sim_matrix = torch.matmul(anchor_feat, cand_feat.t()) / self.temperature  # (B1, B2)
        
        # Tạo mask cho positive pairs (cùng identity)
        mask = torch.eq(
            anchor_label.unsqueeze(1).expand(B1, B2),
            cand_label.unsqueeze(0).expand(B1, B2)
        ).float().to(self.device)  # (B1, B2)
        
        # Eq. 5: Tính loss cho image-to-text
        # Tính log_softmax
        exp_sim = torch.exp(sim_matrix)  # (B1, B2)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + self.eps)  # (B1, B2)
        
        # Chỉ lấy positive pairs (sum over p in P(i))
        # Tính mean log probability cho mỗi anchor
        log_prob_pos = mask * log_prob  # (B1, B2)
        
        # Số lượng positive cho mỗi anchor
        num_pos = mask.sum(dim=1)  # (B1,)
        
        # Tránh chia cho 0: nếu không có positive nào, đặt loss = 0
        num_pos = torch.clamp(num_pos, min=1.0)
        
        # Loss cho từng anchor
        loss_per_anchor = -log_prob_pos.sum(dim=1) / num_pos  # (B1,)
        
        # Tổng loss
        loss = loss_per_anchor.mean()
        
        return loss