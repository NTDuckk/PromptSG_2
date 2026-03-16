import torch

def make_optimizer_1stage(cfg, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        if "prompt_learner" not in key:
            continue

        lr = cfg.SOLVER.STAGE1.BASE_LR
        weight_decay = cfg.SOLVER.STAGE1.WEIGHT_DECAY

        params.append({
            "params": [value],
            "lr": lr,
            "weight_decay": weight_decay,
            "name": key,
            "lr_role": "stage1",
        })

    opt_name = cfg.SOLVER.STAGE1.OPTIMIZER_NAME
    if opt_name == "SGD":
        optimizer = torch.optim.SGD(
            params,
            lr=cfg.SOLVER.STAGE1.BASE_LR,
            momentum=getattr(cfg.SOLVER.STAGE1, "MOMENTUM", 0.9),
        )
    elif opt_name == "Adam":
        optimizer = torch.optim.Adam(
            params,
            lr=cfg.SOLVER.STAGE1.BASE_LR,
            betas=(
                getattr(cfg.SOLVER.STAGE1, "ALPHA", 0.9),
                getattr(cfg.SOLVER.STAGE1, "BETA", 0.999),
            ),
            eps=1e-3,
        )
    elif opt_name == "AdamW":
        optimizer = torch.optim.AdamW(
            params,
            lr=cfg.SOLVER.STAGE1.BASE_LR,
            betas=(
                getattr(cfg.SOLVER.STAGE1, "ALPHA", 0.9),
                getattr(cfg.SOLVER.STAGE1, "BETA", 0.999),
            ),
            eps=1e-8,
        )
    else:
        raise NotImplementedError(f"Unsupported optimizer: {opt_name}")

    return optimizer


def make_optimizer_promptsg(cfg, model, center_criterion):
    params = []

    base_lr = cfg.SOLVER.STAGE2.BASE_LR
    lr_factor = cfg.SOLVER.STAGE2.LR_FACTOR
    bias_lr_factor = cfg.SOLVER.STAGE2.BIAS_LR_FACTOR
    weight_decay = cfg.SOLVER.STAGE2.WEIGHT_DECAY
    weight_decay_bias = cfg.SOLVER.STAGE2.WEIGHT_DECAY_BIAS

    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue

        # text encoder frozen
        if "text_encoder" in key:
            continue

        # default: all non-image_encoder params use large lr
        if "image_encoder" in key:
            lr = base_lr
            lr_role = "image_encoder"
        else:
            lr = base_lr * lr_factor
            lr_role = "other"

        wd = weight_decay

        # bias params: base_lr * bias_factor
        # note: image_encoder bias will still be overridden later by the custom scheduler
        if "bias" in key:
            lr = base_lr * bias_lr_factor
            wd = weight_decay_bias

        params.append({
            "params": [value],
            "lr": lr,
            "weight_decay": wd,
            "name": key,
            "lr_role": lr_role,
        })

    opt_name = cfg.SOLVER.STAGE2.OPTIMIZER_NAME
    if opt_name == "SGD":
        optimizer = torch.optim.SGD(
            params,
            lr=base_lr,
            momentum=getattr(cfg.SOLVER.STAGE2, "MOMENTUM", 0.9),
        )
    elif opt_name == "Adam":
        optimizer = torch.optim.Adam(
            params,
            lr=base_lr,
            betas=(
                getattr(cfg.SOLVER.STAGE2, "ALPHA", 0.9),
                getattr(cfg.SOLVER.STAGE2, "BETA", 0.999),
            ),
            eps=1e-3,
        )
    elif opt_name == "AdamW":
        optimizer = torch.optim.AdamW(
            params,
            lr=base_lr,
            betas=(
                getattr(cfg.SOLVER.STAGE2, "ALPHA", 0.9),
                getattr(cfg.SOLVER.STAGE2, "BETA", 0.999),
            ),
            eps=1e-8,
        )
    else:
        raise NotImplementedError(f"Unsupported optimizer: {opt_name}")

    optimizer_center = torch.optim.SGD(
        center_criterion.parameters(),
        lr=getattr(cfg.SOLVER.STAGE2, "CENTER_LR", 0.5),
    )

    return optimizer, optimizer_center


# giữ tương thích với code cũ
def make_optimizer_2stage(cfg, model, center_criterion):
    return make_optimizer_promptsg(cfg, model, center_criterion)