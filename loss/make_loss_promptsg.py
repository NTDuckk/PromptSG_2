import torch
import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss


def make_loss(cfg, num_classes):
    sampler = cfg.DATALOADER.SAMPLER
    
    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)

    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        def loss_func(score, feat, target, target_cam, image_feat=None, text_feat=None):
            # PromptSG: score có thể là list (từ các tầng khác nhau)
            # feat có thể là list (các feature cho triplet loss)
            # image_feat và text_feat dùng cho SupCon loss
            
            # ID Loss (giống CLIP-ReID)
            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                if isinstance(score, list):
                    ID_LOSS = [xent(scor, target) for scor in score]
                    ID_LOSS = sum(ID_LOSS)
                else:
                    ID_LOSS = xent(score, target)
            else:
                if isinstance(score, list):
                    ID_LOSS = [F.cross_entropy(scor, target) for scor in score]
                    ID_LOSS = sum(ID_LOSS)
                else:
                    ID_LOSS = F.cross_entropy(score, target)

            # Triplet Loss (giống CLIP-ReID) - BỎ WEIGHT
            if isinstance(feat, list):
                TRI_LOSS = [triplet(feats, target)[0] for feats in feat]
                TRI_LOSS = sum(TRI_LOSS)  # Đơn giản tổng các loss, không dùng weight
            else:
                TRI_LOSS = triplet(feat, target)[0]

            # Tính tổng loss cơ bản
            total_loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
            
            # PromptSG: thêm SupCon loss nếu có image_feat và text_feat
            if image_feat is not None and text_feat is not None:
                # Normalize features
                img_norm = F.normalize(image_feat, dim=1)
                txt_norm = F.normalize(text_feat, dim=1)
                
                # Tính supervised contrastive loss
                # image-to-text
                sim_i2t = torch.matmul(img_norm, txt_norm.t()) / cfg.MODEL.PROMPTSG.TEMPERATURE
                # text-to-image  
                sim_t2i = torch.matmul(txt_norm, img_norm.t()) / cfg.MODEL.PROMPTSG.TEMPERATURE
                
                # Tạo mask cho positive pairs (cùng ID)
                batch_size = target.size(0)
                mask = target.expand(batch_size, batch_size).eq(target.expand(batch_size, batch_size).t())
                
                # Tính loss cho cả hai chiều (PromptSG Eq.2-3)
                loss_i2t = -torch.log(
                    (F.softmax(sim_i2t, dim=1) * mask.float()).sum(dim=1) / 
                    F.softmax(sim_i2t, dim=1).sum(dim=1)
                ).mean()
                
                loss_t2i = -torch.log(
                    (F.softmax(sim_t2i, dim=1) * mask.float()).sum(dim=1) / 
                    F.softmax(sim_t2i, dim=1).sum(dim=1)
                ).mean()
                
                SUPCON_LOSS = loss_i2t + loss_t2i
                total_loss += cfg.MODEL.PROMPTSG.LAMBDA_SUPCON * SUPCON_LOSS
                
                return total_loss, {
                    'id_loss': ID_LOSS.detach(),
                    'tri_loss': TRI_LOSS.detach(), 
                    'supcon_loss': SUPCON_LOSS.detach()
                }
            
            return total_loss, {
                'id_loss': ID_LOSS.detach(),
                'tri_loss': TRI_LOSS.detach(),
                'supcon_loss': torch.tensor(0.0).to(ID_LOSS.device)
            }

    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    
    return loss_func