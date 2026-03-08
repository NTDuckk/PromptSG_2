import logging
import os
import time
from datetime import timedelta

import torch
import torch.nn as nn
from torch.cuda import amp

from loss.softmax_loss import CrossEntropyLabelSmooth
from loss.supcontrast import SupConLoss
from loss.triplet_loss import TripletLoss
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _build_losses(cfg, num_classes, device):
    if getattr(cfg.MODEL, "IF_LABELSMOOTH", "off") == "on":
        id_criterion = CrossEntropyLabelSmooth(num_classes=num_classes)
    else:
        id_criterion = nn.CrossEntropyLoss()

    margin = getattr(cfg.SOLVER, "MARGIN", 0.3)
    triplet = TripletLoss(margin=margin)
    supcon = SupConLoss(device)
    return id_criterion, triplet, supcon


def _compute_promptsg_loss(outputs, pids, id_criterion, triplet, supcon, cfg):
    id_loss = id_criterion(outputs["cls_score"], pids)

    tri_loss = 0.0
    for feat in outputs["triplet_feats"]:
        tri_loss = tri_loss + triplet(feat, pids)[0]

    image_feat = torch.nn.functional.normalize(outputs["global_image"], dim=1)
    text_feat = torch.nn.functional.normalize(outputs["text_feat"], dim=1)
    sup_loss = supcon(text_feat, image_feat, pids, pids) + supcon(image_feat, text_feat, pids, pids)

    id_w = getattr(cfg.MODEL, "ID_LOSS_WEIGHT", 1.0)
    tri_w = getattr(cfg.MODEL, "TRIPLET_LOSS_WEIGHT", 1.0)
    sup_w = getattr(cfg.MODEL, "SUPCON_LOSS_WEIGHT", 0.5)

    total = id_w * id_loss + tri_w * tri_loss + sup_w * sup_loss
    return total, id_loss, tri_loss, sup_loss


def do_train_promptsg(
    cfg,
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    num_query,
    num_classes,
    local_rank=0,
    eval_prompt_mode="simplified",
):
    logger = logging.getLogger("transreid.train")
    logger.info("start PromptSG training")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = cfg.SOLVER.MAX_EPOCHS
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    if cfg.OUTPUT_DIR:
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    model.to(device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1 and not hasattr(model, "module"):
        print("Using {} GPUs for training".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    base_model = _unwrap_model(model)
    base_model.set_inference_prompt_mode(eval_prompt_mode)

    loss_meter = AverageMeter()
    id_meter = AverageMeter()
    tri_meter = AverageMeter()
    sup_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler(enabled=torch.cuda.is_available())
    id_criterion, triplet, supcon = _build_losses(cfg, num_classes, device)

    use_sie_camera = getattr(cfg.MODEL, "SIE_CAMERA", False)
    use_sie_view = getattr(cfg.MODEL, "SIE_VIEW", False)

    all_start_time = time.monotonic()
    best_mAP = -1.0
    best_rank1 = -1.0

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        id_meter.reset()
        tri_meter.reset()
        sup_meter.reset()
        acc_meter.reset()
        evaluator.reset()

        model.train()
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
            optimizer.zero_grad()

            img = img.to(device)
            target = vid.to(device)

            if use_sie_camera:
                target_cam = target_cam.to(device)
            else:
                target_cam = None

            if use_sie_view:
                target_view = target_view.to(device)
            else:
                target_view = None

            with amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(x=img, label=target, cam_label=target_cam, view_label=target_view)
                loss, id_loss, tri_loss, sup_loss = _compute_promptsg_loss(
                    outputs, target, id_criterion, triplet, supcon, cfg
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                acc = (outputs["cls_score"].max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            id_meter.update(id_loss.item(), img.shape[0])
            tri_meter.update(float(tri_loss.item() if hasattr(tri_loss, "item") else tri_loss), img.shape[0])
            sup_meter.update(sup_loss.item(), img.shape[0])
            acc_meter.update(acc.item(), 1)

            if (n_iter + 1) % log_period == 0:
                current_lrs = [group["lr"] for group in optimizer.param_groups]
                vis_lr = current_lrs[0]
                new_lr = current_lrs[1] if len(current_lrs) > 1 else current_lrs[0]
                logger.info(
                    "Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, ID: {:.3f}, TRI: {:.3f}, "
                    "SupCon: {:.3f}, Acc: {:.3f}, Lr(vis/new): {:.2e}/{:.2e}".format(
                        epoch,
                        n_iter + 1,
                        len(train_loader),
                        loss_meter.avg,
                        id_meter.avg,
                        tri_meter.avg,
                        sup_meter.avg,
                        acc_meter.avg,
                        vis_lr,
                        new_lr,
                    )
                )

        scheduler.step()

        end_time = time.time()
        time_per_batch = (end_time - start_time) / max(1, (n_iter + 1))
        logger.info(
            "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]".format(
                epoch,
                time_per_batch,
                train_loader.batch_size / max(time_per_batch, 1e-12),
            )
        )

        if epoch % checkpoint_period == 0:
            ckpt_name = "{}_promptsg_epoch{}.pth".format(cfg.MODEL.NAME, epoch)
            ckpt_path = os.path.join(cfg.OUTPUT_DIR, ckpt_name) if cfg.OUTPUT_DIR else ckpt_name
            torch.save(_unwrap_model(model).state_dict(), ckpt_path)
            logger.info("Saved checkpoint to {}".format(ckpt_path))

        if epoch % eval_period == 0:
            rank1, rank5, mAP = do_inference_promptsg(
                cfg=cfg,
                model=model,
                val_loader=val_loader,
                num_query=num_query,
                prompt_mode=eval_prompt_mode,
                logger_name="transreid.train",
                save_result=False,
            )
            logger.info(
                "Validation Results - Epoch: {} | prompt_mode={} | mAP: {:.1%} | "
                "Rank-1: {:.1%} | Rank-5: {:.1%}".format(
                    epoch, eval_prompt_mode, mAP, rank1, rank5
                )
            )
            if mAP > best_mAP:
                best_mAP = mAP
                best_rank1 = rank1
                best_name = "{}_promptsg_best.pth".format(cfg.MODEL.NAME)
                best_path = os.path.join(cfg.OUTPUT_DIR, best_name) if cfg.OUTPUT_DIR else best_name
                torch.save(_unwrap_model(model).state_dict(), best_path)
                logger.info("Saved new best checkpoint to {}".format(best_path))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Total PromptSG running time: {}".format(total_time))
    logger.info("Best mAP: {:.1%}, Best Rank-1: {:.1%}".format(best_mAP, best_rank1))


def do_inference_promptsg(
    cfg,
    model,
    val_loader,
    num_query,
    prompt_mode="simplified",
    logger_name="transreid.test",
    save_result=True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = logging.getLogger(logger_name)
    logger.info("Enter PromptSG inferencing with prompt_mode={}".format(prompt_mode))

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator.reset()

    if torch.cuda.is_available() and torch.cuda.device_count() > 1 and not hasattr(model, "module"):
        print("Using {} GPUs for inference".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    model.to(device)

    base_model = _unwrap_model(model)
    base_model.set_inference_prompt_mode(prompt_mode)

    use_sie_camera = getattr(cfg.MODEL, "SIE_CAMERA", False)
    use_sie_view = getattr(cfg.MODEL, "SIE_VIEW", False)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)

            if use_sie_camera:
                camids = camids.to(device)
            else:
                camids = None

            if use_sie_view:
                target_view = target_view.to(device)
            else:
                target_view = None

            feat = model(x=img, cam_label=camids, view_label=target_view, prompt_mode=prompt_mode)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

    return cmc[0], cmc[4], mAP