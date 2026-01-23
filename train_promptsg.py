import argparse
import os
import random

import numpy as np
import torch

from config import cfg
from datasets.make_dataloader_promptsg import make_dataloader
from loss.make_loss_promptsg import make_loss
from model.make_model_promptsg import make_model
from solver.lr_scheduler import WarmupMultiStepLR
from solver.make_optimizer_promptsg import make_optimizer
from utils.logger import setup_logger
from processor.processor_promptsg import do_train


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PromptSG Training')
    parser.add_argument('--config_file', default='configs/person/vit_promptsg.yml', type=str)
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    parser.add_argument('--local_rank', default=0, type=int)
    args = parser.parse_args()

    if args.config_file != '':
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger('promptsg', output_dir, if_train=True)
    logger.info('Saving model in the path :{}'.format(cfg.OUTPUT_DIR))
    logger.info(args)

    if args.config_file != '':
        logger.info('Loaded configuration file {}'.format(args.config_file))
        with open(args.config_file, 'r') as cf:
            logger.info('\n' + cf.read())
    logger.info('Running with config:\n{}'.format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    train_loader, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)

    loss_fn = make_loss(cfg, num_classes=num_classes)

    optimizer = make_optimizer(cfg, model)

    scheduler = WarmupMultiStepLR(
        optimizer,
        cfg.SOLVER.PROMPTSG.STEPS,
        cfg.SOLVER.PROMPTSG.GAMMA,
        warmup_factor=1.0,
        warmup_iters=0,
        warmup_method='linear',
    )

    do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
        num_query,
        args.local_rank,
    )
