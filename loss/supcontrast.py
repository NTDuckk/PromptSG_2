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

import torch
import torch.nn as nn
import torch.nn.functional as F

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

class SupConLoss(torch.nn.Module):
    def __init__(self, temperature=0.07, eps=1e-12, normalize=True):
        super().__init__()
        self.temperature = temperature
        self.eps = eps
        self.normalize = normalize
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, anchor_feat, cand_feat, anchor_label, cand_label):
        """
        Supervised Contrastive Loss (Equations 5-6 in PromptSG paper)
        
        Args:
            anchor_feat: (B1, D) - features của anchor
            cand_feat: (B2, D) - features của candidate  
            anchor_label: (B1,) - labels của anchor
            cand_label: (B2,) - labels của candidate
        """
        if self.normalize:
            anchor_feat = F.normalize(anchor_feat, dim=1)
            cand_feat = F.normalize(cand_feat, dim=1)

        B1, B2 = anchor_feat.size(0), cand_feat.size(0)
        
        # Similarity matrix
        sim_matrix = torch.matmul(anchor_feat, cand_feat.t()) / self.temperature  # (B1, B2)
        
        # Positive mask: same identity
        mask = torch.eq(
            anchor_label.unsqueeze(1).expand(B1, B2),
            cand_label.unsqueeze(0).expand(B1, B2)
        ).float().to(self.device)  # (B1, B2)
        
        # Log softmax over candidates
        logits = F.log_softmax(sim_matrix, dim=1)  # (B1, B2)
        
        # Loss: - sum over positive pairs, then average per anchor
        loss_pos = - (mask * logits).sum(dim=1)  # (B1,) sum over positives
        
        # Normalize by number of positives (avoid division by zero)
        num_pos = mask.sum(dim=1).clamp(min=1.0)
        loss_per_anchor = loss_pos / num_pos
        
        # Average over all anchors
        loss = loss_per_anchor.mean()
        
        return loss


def symmetric_supervised_contrastive_loss(v_features, l_features, labels, temperature=0.07, eps=1e-12):
    """
    Symmetric Supervised Contrastive Loss (Equation 4-5 in PromptSG paper)
    
    Args:
        v_features: visual features (B, D)
        l_features: text features (B, D)
        labels: identity labels (B,)
        temperature: temperature parameter τ
        eps: small value for numerical stability
    """
    batch_size = v_features.size(0)
    
    # Normalize features (cosine similarity)
    v_norm = F.normalize(v_features, dim=1)  # (B, D)
    l_norm = F.normalize(l_features, dim=1)  # (B, D)
    
    # Create positive mask: 1 for same identity, 0 otherwise
    labels = labels.view(-1, 1)  # (B, 1)
    mask = torch.eq(labels, labels.T).float().to(v_features.device)  # (B, B)
    
    # Image-to-Text loss (Eq. 5, first part)
    sim_i2t = torch.matmul(v_norm, l_norm.T) / temperature  # (B, B)
    
    # Log softmax with positive pairs
    logits_i2t = F.log_softmax(sim_i2t, dim=1)  # (B, B)
    
    # Only sum over positive pairs (P(i))
    loss_i2t = - (mask * logits_i2t).sum(dim=1)  # (B,)
    
    # Normalize by number of positive pairs (avoid division by zero)
    num_pos = mask.sum(dim=1)  # (B,)
    num_pos = torch.clamp(num_pos, min=1.0)
    loss_i2t = loss_i2t / num_pos
    loss_i2t = loss_i2t.mean()
    
    # Text-to-Image loss (Eq. 5, second part)
    sim_t2i = torch.matmul(l_norm, v_norm.T) / temperature  # (B, B)
    logits_t2i = F.log_softmax(sim_t2i, dim=1)  # (B, B)
    
    loss_t2i = - (mask * logits_t2i).sum(dim=1)  # (B,)
    loss_t2i = loss_t2i / num_pos
    loss_t2i = loss_t2i.mean()
    
    # Total loss (Eq. 4)
    loss = (loss_i2t + loss_t2i)
    
    return loss