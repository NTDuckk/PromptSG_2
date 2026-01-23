import logging
import os
import time
import torch
import torch.nn as nn
from torch.cuda import amp
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
import torch.distributed as dist
from torch.nn import functional as F

def do_train(cfg, model, train_loader, val_loader, optimizer, scheduler, loss_fn, num_query, local_rank):
    log_period = cfg.SOLVER.PROMPTSG.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.PROMPTSG.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.PROMPTSG.EVAL_PERIOD
    epochs = cfg.SOLVER.PROMPTSG.MAX_EPOCHS

    device = cfg.MODEL.DEVICE

    logger = logging.getLogger("promptsg.train")
    logger.info('start training')
    logger.info("Config:\n{}".format(cfg.dump()))

    if device:
        device = torch.device(f"cuda:{local_rank}") if local_rank is not None else torch.device("cuda")
        model.to(device)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    id_meter = AverageMeter()
    tri_meter = AverageMeter()
    supcon_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()

    all_start_time = time.monotonic()

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset(); acc_meter.reset(); evaluator.reset()
        id_meter.reset(); tri_meter.reset(); supcon_meter.reset()

        logger.info("Epoch {} started".format(epoch))
        scheduler.step()
        model.train()

        for n_iter, (img, pid, camid, viewid) in enumerate(train_loader):
            optimizer.zero_grad()
            img = img.to(device)
            target = pid.to(device)

            with amp.autocast(enabled=True):
                cls_score, triplet_feats, image_feat, text_feat = model(img, target)
                loss, id_loss, tri_loss, supcon_loss = loss_fn(cls_score, triplet_feats, image_feat, text_feat, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                acc = (cls_score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            id_meter.update(id_loss.item(), img.shape[0])
            tri_meter.update(tri_loss.item(), img.shape[0])
            supcon_meter.update(supcon_loss.item(), img.shape[0])
            acc_meter.update(acc.item(), 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info(
                    "Epoch[{}] Iteration[{}/{}] Loss: {:.3f} (ID {:.3f} TRI {:.3f} SupCon {:.3f}) Acc: {:.3f} Lr: {:.2e}".format(
                        epoch, n_iter + 1, len(train_loader),
                        loss_meter.avg, id_meter.avg, tri_meter.avg, supcon_meter.avg,
                        acc_meter.avg,
                        scheduler.get_lr()[0] if hasattr(scheduler, 'get_lr') else optimizer.param_groups[0]['lr']
                    )
                )
        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]".format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, pid, camid, camids_batch, viewid, img_path) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            feat = model(img)
                            evaluator.update((feat, pid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
                evaluator.reset()
                for n_iter, (img, pid, camid, camids_batch, viewid, img_path) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        feat = model(img)
                        evaluator.update((feat, pid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()

    total_time = time.monotonic() - all_start_time
    logger.info("Total running time: {:.1f}[s]".format(total_time))


def do_inference(cfg, model, val_loader, num_query):
    device = cfg.MODEL.DEVICE
    logger = logging.getLogger("promptsg.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()

    for n_iter, (img, pid, camid, camid_batch, viewid, img_path) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            feat = model(img)
            evaluator.update((feat, pid, camid))

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4], mAP
