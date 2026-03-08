import argparse
import os
import random

import numpy as np
import torch

from config import cfg
from datasets.make_dataloader_clipreid import make_dataloader
from model.make_model_promptsg import make_model
from processor.processor_promptsg import do_inference_promptsg, do_train_promptsg
from utils.logger import setup_logger


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, milestones, gamma=0.1, warmup_epochs=0, warmup_factor=0.1, last_epoch=-1):
        self.milestones = sorted(milestones)
        self.gamma = gamma
        self.warmup_epochs = warmup_epochs
        self.warmup_factor = warmup_factor
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch
        warmup_multiplier = 1.0
        if self.warmup_epochs > 0 and epoch < self.warmup_epochs:
            alpha = float(epoch + 1) / float(max(1, self.warmup_epochs))
            warmup_multiplier = self.warmup_factor * (1.0 - alpha) + alpha

        decay_count = 0
        for milestone in self.milestones:
            if epoch >= milestone:
                decay_count += 1
        decay_multiplier = self.gamma ** decay_count

        return [base_lr * warmup_multiplier * decay_multiplier for base_lr in self.base_lrs]


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def build_optimizer(cfg, model, visual_lr=None, new_module_lr=None):
    weight_decay = cfg.SOLVER.WEIGHT_DECAY

    if visual_lr is None:
        visual_lr = cfg.SOLVER.VISUAL_BASE_LR
    if new_module_lr is None:
        new_module_lr = cfg.SOLVER.NEW_MODULE_BASE_LR

    param_groups = model.get_param_groups(
        visual_lr=visual_lr,
        new_lr=new_module_lr,
        weight_decay=weight_decay,
    )
    optimizer = torch.optim.Adam(param_groups)
    return optimizer


def main():
    parser = argparse.ArgumentParser(description='PromptSG-style ReID training on top of the current CLIP-ReID repo')
    parser.add_argument('--config_file', default='configs/person/vit_clipreid.yml', type=str)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--eval_prompt_mode', default='simplified', choices=['simplified', 'composed'])
    parser.add_argument('--visual_lr', default=None, type=float)
    parser.add_argument('--new_module_lr', default=None, type=float)
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger('transreid', output_dir, if_train=not args.eval_only)
    logger.info('Saving model in the path :{}'.format(cfg.OUTPUT_DIR))
    logger.info(args)
    if args.config_file:
        logger.info('Loaded configuration file {}'.format(args.config_file))
        with open(args.config_file, 'r') as cf:
            logger.info('\n' + cf.read())
    logger.info('Running with config:\n{}'.format(cfg))

    train_loader_stage2, _, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)

    if args.resume:
        model.load_param(args.resume)

    if args.eval_only:
        do_inference_promptsg(
            cfg=cfg,
            model=model,
            val_loader=val_loader,
            num_query=num_query,
            prompt_mode=args.eval_prompt_mode,
        )
        return

    optimizer = build_optimizer(
        cfg=cfg,
        model=model,
        visual_lr=args.visual_lr,
        new_module_lr=args.new_module_lr,
    )

    warmup_epochs = cfg.SOLVER.WARMUP_EPOCHS
    scheduler = WarmupMultiStepLR(
        optimizer,
        milestones=list(cfg.SOLVER.STEPS),
        gamma=cfg.SOLVER.GAMMA,
        warmup_epochs=warmup_epochs,
        warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
    )

    do_train_promptsg(
        cfg=cfg,
        model=model,
        train_loader=train_loader_stage2,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_query=num_query,
        num_classes=num_classes,
        local_rank=args.local_rank,
        eval_prompt_mode=args.eval_prompt_mode,
    )


if __name__ == '__main__':
    main()
