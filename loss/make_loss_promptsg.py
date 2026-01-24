import torch
import torch.nn.functional as F

from .softmax_loss import CrossEntropyLabelSmooth
from .triplet_loss import TripletLoss
from .supcontrast import SupConLoss


def make_loss(cfg, num_classes):
    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
    else:
        xent = None

    if cfg.MODEL.NO_MARGIN:
        triplet = TripletLoss()
    else:
        triplet = TripletLoss(cfg.SOLVER.MARGIN)

    supcon = SupConLoss(device=cfg.MODEL.DEVICE, temperature=cfg.MODEL.PROMPTSG.TEMPERATURE)

    def loss_func(cls_score, triplet_feats, image_feat, text_feat, target):
        if xent is not None:
            id_loss = xent(cls_score, target)
        else:
            id_loss = F.cross_entropy(cls_score, target)

        if isinstance(triplet_feats, (list, tuple)):
            # tri_loss = sum(triplet(f, target)[0] for f in triplet_feats)
            weights = [0.3, 0.3, 0.4]
            tri_loss = sum(w * triplet(f, target)[0] for w, f in zip(weights, triplet_feats))
        else:
            tri_loss = triplet(triplet_feats, target)[0]

        img_n = F.normalize(image_feat, dim=1)
        txt_n = F.normalize(text_feat, dim=1)

        loss_i2t = supcon(img_n, txt_n, target, target)
        loss_t2i = supcon(txt_n, img_n, target, target)
        supcon_loss = loss_i2t + loss_t2i

        total = tri_loss + id_loss + cfg.MODEL.PROMPTSG.LAMBDA_SUPCON * supcon_loss
        return total, id_loss.detach(), tri_loss.detach(), supcon_loss.detach()

    return loss_func
