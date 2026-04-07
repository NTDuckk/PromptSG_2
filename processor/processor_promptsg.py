import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist


def do_train_promptsg(cfg,
                      model,
                      center_criterion,
                      train_loader,
                      val_loader,
                      optimizer,
                      optimizer_center,
                      scheduler,
                      loss_fn,
                      num_query,
                      query_loader, 
                      gallery_loader,
                      local_rank):
    log_period = cfg.SOLVER.STAGE2.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.STAGE2.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.STAGE2.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.STAGE2.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info("start training")

    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)

    total_loss_meter = AverageMeter()
    supcon_loss_meter = AverageMeter()
    id_loss_meter = AverageMeter()
    tri_loss_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()

    from datetime import timedelta
    all_start_time = time.monotonic()

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        total_loss_meter.reset()
        supcon_loss_meter.reset()
        id_loss_meter.reset()
        tri_loss_meter.reset()
        evaluator.reset()
        evaluator.reset_gallery()
        
        scheduler.step()
        model.train()

        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()

            img = img.to(device)
            target = vid.to(device)

            if cfg.MODEL.SIE_CAMERA:
                target_cam = target_cam.to(device)
            else:
                target_cam = None

            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else:
                target_view = None

            with amp.autocast(enabled=True):
                outputs = model(
                    x=img,
                    label=target,
                    cam_label=target_cam,
                    view_label=target_view
                )

                loss_dict = loss_fn(outputs, target, target_cam)

                supcon_loss = loss_dict["supcon_loss"]
                id_loss = loss_dict["id_loss"]
                tri_loss = loss_dict["tri_loss"]

                total_loss = (
                    cfg.MODEL.I2T_LOSS_WEIGHT * supcon_loss
                    + cfg.MODEL.ID_LOSS_WEIGHT * id_loss
                    + cfg.MODEL.TRIPLET_LOSS_WEIGHT * tri_loss
                )

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    if param.grad is not None:
                        param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()

            bs = img.shape[0]
            total_loss_meter.update(total_loss.item(), bs)
            supcon_loss_meter.update(supcon_loss.item(), bs)
            id_loss_meter.update(id_loss.item(), bs)
            tri_loss_meter.update(tri_loss.item(), bs)
            
            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info(
                    "Epoch[{}] Iteration[{}/{}] "
                    "Total: {:.3f}, SupCon: {:.3f}, ID: {:.3f}, TRI: {:.3f}, Base Lr: {:.2e}".format(
                        epoch,
                        n_iter + 1,
                        len(train_loader),
                        total_loss_meter.avg,
                        supcon_loss_meter.avg,
                        id_loss_meter.avg,
                        tri_loss_meter.avg,
                        scheduler.get_lr()[0]
                    )
                )

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)

        if not cfg.MODEL.DIST_TRAIN:
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]".format(
                    epoch, time_per_batch, train_loader.batch_size / time_per_batch
                )
            )
            logger.info(
                "Epoch {} summary - Total: {:.3f}, SupCon: {:.3f}, ID: {:.3f}, TRI: {:.3f}".format(
                    epoch,
                    total_loss_meter.avg,
                    supcon_loss_meter.avg,
                    id_loss_meter.avg,
                    tri_loss_meter.avg
                )
            )

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(
                        model.state_dict(),
                        os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch))
                    )
            else:
                torch.save(
                    model.state_dict(),
                    os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch))
                )

        eval_mode = cfg.TEST.EVAL_MODE
        loader_flag = cfg.DATASETS.DATASET_FLAG
        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            if cfg.MODEL.SIE_CAMERA:
                                camids = camids.to(device)
                            else:
                                camids = None
                            if cfg.MODEL.SIE_VIEW:
                                target_view = target_view.to(device)
                            else:
                                target_view = None
                            feat = model(img, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, vid, camid))

                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                print('Model not dist_train')
                if eval_mode == 'clipreid':
                    print('Eval mode: clipreid')
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            if cfg.MODEL.SIE_CAMERA:
                                camids = camids.to(device)
                            else:
                                camids = None
                            if cfg.MODEL.SIE_VIEW:
                                target_view = target_view.to(device)
                            else:
                                target_view = None
                            feat = model(img, cam_label=camids, view_label=target_view, eval_mode = eval_mode)
                            evaluator.update((feat, vid, camid))

                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
                elif eval_mode == 'cross_cls':
                    print('Eval mode: cross_cls')
                    loader1 = query_loader
                    loader2 = gallery_loader
                    if loader_flag == 'clipreid':
                        loader1 = val_loader
                        loader2 = val_loader
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(loader1):
                        with torch.no_grad():
                            img = img.to(device)
                            if cfg.MODEL.SIE_CAMERA:
                                camids = camids.to(device)
                            else:
                                camids = None
                            if cfg.MODEL.SIE_VIEW:
                                target_view = target_view.to(device)
                            else:
                                target_view = None
                            feat_query = model(img, cam_label=camids, view_label=target_view,
                                               eval_mode = eval_mode, dataset_flag = 'query')
                            evaluator.update((feat_query, vid, camid))
                    
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(loader2):
                        with torch.no_grad():
                            img = img.to(device)
                            if cfg.MODEL.SIE_CAMERA:
                                camids = camids.to(device)
                            else:
                                camids = None
                            if cfg.MODEL.SIE_VIEW:
                                target_view = target_view.to(device)
                            else:
                                target_view = None
                            feat_gallery = model(img, cam_label=camids, view_label=target_view,
                                               eval_mode = eval_mode, dataset_flag = 'gallery')
                            evaluator.update_gallery((feat_gallery, vid, camid))
                    
                    cmc, mAP, _, _, _, _, _, _, _, _, _ = evaluator.compute_cross_cls()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP / 100.0))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1] / 100.0))
                    torch.cuda.empty_cache()

    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Total running time: {}".format(total_time))
    print(cfg.OUTPUT_DIR)


def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator.reset()
    evaluator.reset_gallery()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            if cfg.MODEL.SIE_CAMERA:
                camids = camids.to(device)
            else:
                camids = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else:
                target_view = None

            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]