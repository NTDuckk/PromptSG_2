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
import subprocess
import sys

def setup_training_logger(cfg):
    """Setup additional file logger for training metrics"""
    # Create logs directory if not exists
    log_dir = cfg.OUTPUT_DIR
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup file logger for metrics
    metrics_logger = logging.getLogger("promptsg.metrics")
    metrics_logger.setLevel(logging.INFO)
    
    # File handler for metrics
    metrics_file = os.path.join(log_dir, 'training_metrics.txt')
    file_handler = logging.FileHandler(metrics_file, mode='w')
    file_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add handler to logger
    if not metrics_logger.handlers:
        metrics_logger.addHandler(file_handler)
    
    return metrics_logger

def auto_generate_plots(cfg):
    """Automatically generate learning curves after training completion"""
    logger = logging.getLogger("promptsg.train")
    logger.info("Generating learning curves...")
    
    try:
        # Get the script directory and plot script path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        plot_script = os.path.join(project_root, 'plot_learning_curves.py')
        
        if not os.path.exists(plot_script):
            logger.warning(f"Plot script not found at {plot_script}")
            return False
        
        # Prepare command
        cmd = [
            sys.executable,  # Use current Python interpreter
            plot_script,
            '--log_dir', cfg.OUTPUT_DIR,
            '--output_dir', os.path.join(project_root, 'plots'),
            '--save_json'
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        # Run the plot script
        result = subprocess.run(
            cmd,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes timeout
        )
        
        if result.returncode == 0:
            logger.info("Learning curves generated successfully!")
            logger.info(f"Output: {result.stdout}")
            if result.stderr:
                logger.info(f"Warnings: {result.stderr}")
            return True
        else:
            logger.error(f"Failed to generate learning curves: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("Plot generation timed out after 5 minutes")
        return False
    except Exception as e:
        logger.error(f"Error generating plots: {str(e)}")
        return False

def do_train(cfg, model, train_loader, val_loader, optimizer, scheduler, loss_fn, num_query, local_rank):
    log_period = cfg.SOLVER.PROMPTSG.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.PROMPTSG.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.PROMPTSG.EVAL_PERIOD
    epochs = cfg.SOLVER.PROMPTSG.MAX_EPOCHS

    device = cfg.MODEL.DEVICE

    logger = logging.getLogger("promptsg.train")
    logger.info('start training')
    logger.info("Config:\n{}".format(cfg.dump()))

    # Setup metrics file logger
    metrics_logger = setup_training_logger(cfg)
    metrics_logger.info("=== TRAINING STARTED ===")
    metrics_logger.info(f"Model: {cfg.MODEL.NAME}")
    metrics_logger.info(f"Prompt mode: {cfg.MODEL.PROMPTSG.PROMPT_MODE}")
    metrics_logger.info(f"Max epochs: {epochs}")
    metrics_logger.info(f"Learning rate: {cfg.SOLVER.PROMPTSG.BASE_LR_VISUAL}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    metrics_logger.info(f"Total parameters: {total_params:,}")
    metrics_logger.info(f"Trainable parameters: {trainable_params:,}")
    metrics_logger.info("="*50)

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
        metrics_logger.info(f"EPOCH {epoch} - Training started")
        model.train()

        for n_iter, (img, pid, camid, viewid) in enumerate(train_loader):
            optimizer.zero_grad()
            img = img.to(device)
            target = pid.to(device)

            with amp.autocast(enabled=True):
                # cls_score, triplet_feats, image_feat, text_feat = model(img, target)
                cls_score, triplet_feats, image_feat, text_feat = model(x = img, label = target)

                # ============ DEBUG: Check model outputs ============
                if epoch == 1 and n_iter == 0:
                    logger.info("=== DEBUG: Model Outputs ===")
                    if isinstance(cls_score, (list, tuple)):
                        logger.info(f"Number of cls_scores: {len(cls_score)}")
                        for i, score in enumerate(cls_score):
                            logger.info(f"  cls_score[{i}] shape: {score.shape}")
                    else:
                        logger.info(f"cls_score shape: {cls_score.shape}")
                    logger.info(f"triplet_feats type: {type(triplet_feats)}")
                    if isinstance(triplet_feats, (list, tuple)):
                        logger.info(f"Number of triplet features: {len(triplet_feats)}")
                        for i, feat in enumerate(triplet_feats):
                            logger.info(f"  triplet_feats[{i}] shape: {feat.shape}")
                            logger.info(f"  triplet_feats[{i}] min/max: {feat.min().item():.4f}/{feat.max().item():.4f}")
                    else:
                        logger.info(f"triplet_feats shape: {triplet_feats.shape}")
                    logger.info(f"image_feat shape: {image_feat.shape}")
                    logger.info(f"text_feat shape: {text_feat.shape}")
                    logger.info(f"target shape: {target.shape}")
                    logger.info(f"Batch size: {img.shape[0]}")
                    logger.info("===========================")
                # ============ END DEBUG ============

                total_loss, losses_dict = loss_fn(cls_score, triplet_feats, target, camid, image_feat, text_feat)
                loss = total_loss
                id_loss = losses_dict['id_loss']
                tri_loss = losses_dict['tri_loss']
                supcon_loss = losses_dict['supcon_loss']
                
                # ============ DEBUG: Check loss values ============
                if epoch == 1 and n_iter == 0:
                    logger.info("=== DEBUG: Loss Values ===")
                    logger.info(f"Total loss: {loss.item():.6f}")
                    logger.info(f"ID loss: {id_loss.item():.6f}")
                    logger.info(f"Triplet loss: {tri_loss.item():.6f}")
                    logger.info(f"SupCon loss: {supcon_loss.item():.6f}")
                    logger.info(f"Lambda SupCon: {cfg.MODEL.PROMPTSG.LAMBDA_SUPCON if hasattr(cfg.MODEL.PROMPTSG, 'LAMBDA_SUPCON') else 'N/A'}")
                    logger.info(f"Lambda Triplet: {cfg.MODEL.PROMPTSG.LAMBDA_TRIPLET if hasattr(cfg.MODEL.PROMPTSG, 'LAMBDA_TRIPLET') else 'N/A'}")
                    logger.info(f"Lambda ID: {cfg.MODEL.PROMPTSG.LAMBDA_ID if hasattr(cfg.MODEL.PROMPTSG, 'LAMBDA_ID') else 'N/A'}")
                    logger.info("==========================")
                # ============ END DEBUG ============

            scaler.scale(loss).backward()

            # Gradient clipping for stability
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                # cls_score is a list [cls_score, cls_score_proj], use first one for accuracy
                main_cls_score = cls_score[0] if isinstance(cls_score, (list, tuple)) else cls_score
                acc = (main_cls_score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            id_meter.update(id_loss.item(), img.shape[0])
            tri_meter.update(tri_loss.item(), img.shape[0])
            supcon_meter.update(supcon_loss.item(), img.shape[0])
            acc_meter.update(acc.item(), 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                log_msg = ("Epoch[{}] Iteration[{}/{}] Loss: {:.3f} (ID {:.3f} TRI {:.3f} SupCon {:.3f}) Acc: {:.3f} Lr: {:.2e}".format(
                    epoch, n_iter + 1, len(train_loader),
                    loss_meter.avg, id_meter.avg, tri_meter.avg, supcon_meter.avg,
                    acc_meter.avg,
                    scheduler.get_lr()[0] if hasattr(scheduler, 'get_lr') else optimizer.param_groups[0]['lr']
                ))
                logger.info(log_msg)
                metrics_logger.info(log_msg)
        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            epoch_msg = "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]".format(epoch, time_per_batch, train_loader.batch_size / time_per_batch)
            logger.info(epoch_msg)
            metrics_logger.info(epoch_msg)

        # Step scheduler AFTER epoch (not before)
        scheduler.step()

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
                    val_msg = "Validation Results - Epoch {}".format(epoch)
                    logger.info(val_msg)
                    metrics_logger.info(val_msg)
                    map_msg = "mAP: {:.1%}".format(mAP)
                    logger.info(map_msg)
                    metrics_logger.info(map_msg)
                    for r in [1, 5, 10]:
                        rank_msg = "Rank-{:<3}:{:.1%}".format(r, cmc[r - 1])
                        logger.info(rank_msg)
                        metrics_logger.info(rank_msg)
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
                val_msg = "Validation Results - Epoch {}".format(epoch)
                logger.info(val_msg)
                metrics_logger.info(val_msg)
                map_msg = "mAP: {:.1%}".format(mAP)
                logger.info(map_msg)
                metrics_logger.info(map_msg)
                for r in [1, 5, 10]:
                    rank_msg = "Rank-{:<3}:{:.1%}".format(r, cmc[r - 1])
                    logger.info(rank_msg)
                    metrics_logger.info(rank_msg)
                torch.cuda.empty_cache()

    total_time = time.monotonic() - all_start_time
    time_msg = "Total running time: {:.1f}[s]".format(total_time)
    logger.info(time_msg)
    metrics_logger.info("="*50)
    metrics_logger.info("=== TRAINING COMPLETED ===")
    metrics_logger.info(time_msg)
    metrics_logger.info("="*50)

    # Auto-generate learning curves
    if not cfg.MODEL.DIST_TRAIN or (cfg.MODEL.DIST_TRAIN and dist.get_rank() == 0):
        logger.info("Training completed. Auto-generating learning curves...")
        success = auto_generate_plots(cfg)
        if success:
            logger.info(" Training completed successfully! Check 'plots/' directory for learning curves.")
        else:
            logger.warning("  Plot generation failed. You can manually run: python plot_learning_curves.py")
    else:
        logger.info("Training completed. Plot generation skipped for distributed training.")

    return


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