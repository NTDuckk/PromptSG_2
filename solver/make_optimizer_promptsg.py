import torch


def make_optimizer(cfg, model):
    params_visual = []
    params_new = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith('image_encoder.'):
            params_visual.append(p)
        else:
            params_new.append(p)

    optim_name = cfg.SOLVER.PROMPTSG.OPTIMIZER_NAME
    wd = cfg.SOLVER.PROMPTSG.WEIGHT_DECAY

    if optim_name == 'Adam':
        optimizer = torch.optim.Adam(
            [
                {"params": params_visual, "lr": cfg.SOLVER.PROMPTSG.BASE_LR_VISUAL, "weight_decay": wd},
                {"params": params_new, "lr": cfg.SOLVER.PROMPTSG.BASE_LR_NEW, "weight_decay": wd},
            ]
        )
    elif optim_name == 'AdamW':
        optimizer = torch.optim.AdamW(
            [
                {"params": params_visual, "lr": cfg.SOLVER.PROMPTSG.BASE_LR_VISUAL, "weight_decay": wd},
                {"params": params_new, "lr": cfg.SOLVER.PROMPTSG.BASE_LR_NEW, "weight_decay": wd},
            ]
        )
    else:
        optimizer = getattr(torch.optim, optim_name)(
            [
                {"params": params_visual, "lr": cfg.SOLVER.PROMPTSG.BASE_LR_VISUAL, "weight_decay": wd},
                {"params": params_new, "lr": cfg.SOLVER.PROMPTSG.BASE_LR_NEW, "weight_decay": wd},
            ]
        )

    return optimizer
