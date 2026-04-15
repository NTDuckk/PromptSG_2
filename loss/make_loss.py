# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss
from .supcontrast import SupConLoss


def compute_TAL(image_features, text_features, pid, tau=0.015, margin=0.1):
    # # normalized features
    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)
    scores = text_norm @ image_norm.t()

    batch_size = scores.shape[0]
    pid = pid.reshape((batch_size, 1))  # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float().cuda()
    mask = 1 - labels

    alpha_i2t = ((scores / tau).exp() * labels / ((scores / tau).exp() * labels).sum(dim=1, keepdim=True)).detach()
    alpha_t2i = ((scores.t() / tau).exp() * labels / ((scores.t() / tau).exp() * labels).sum(dim=1,
                                                                                             keepdim=True)).detach()

    loss = (-  (alpha_i2t * scores).sum(1) + tau * ((scores / tau).exp() * mask).sum(1).clamp(max=10e35).log() + margin).clamp(min=0) \
           + (-  (alpha_t2i * scores.t()).sum(1) + tau * ((scores.t() / tau).exp() * mask).sum(1).clamp(max=10e35).log() + margin).clamp(min=0)

    return loss.sum()


def make_loss(cfg, num_classes):    # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    feat_dim = 2048
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    device = "cuda"
    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))
    supcon = SupConLoss(device)
    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)
    
    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        # def loss_func(score, feat, target, target_cam, i2tscore = None):
        def loss_func(outputs, target, target_cam=None):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                score = outputs["cls_score"]
                feat = outputs["global_feat"]
                text_feat = outputs["text_feat"]
                img_feat_proj = outputs["img_feat_proj"]
                
                 # ID loss
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    if isinstance(score, list):
                        id_loss = sum([xent(s, target) for s in score])
                    else:
                        id_loss = xent(score, target)
                else:
                    if isinstance(score, list):
                        id_loss = sum([F.cross_entropy(s, target) for s in score])
                    else:
                        id_loss = F.cross_entropy(score, target)

                # Triplet loss
                if isinstance(feat, list):
                    tri_loss = sum([triplet(f, target)[0] for f in feat])
                else:
                    tri_loss = triplet(feat, target)[0]
                
                cross_feat = feat[0]
                TAL_loss = compute_TAL(cross_feat, img_feat_proj, target)
                tri_loss += TAL_loss
                
                # Symmetric SupCon
                supcon_i2t = supcon(text_feat, img_feat_proj, target, target)
                supcon_t2i = supcon(img_feat_proj, text_feat, target, target)
                supcon_loss = supcon_i2t + supcon_t2i

                return {
                    "supcon_loss": supcon_loss,
                    "id_loss": id_loss,
                    "tri_loss": tri_loss,
                }

            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func, center_criterion


