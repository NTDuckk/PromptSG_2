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
    keys = []
    for key, value in model.named_parameters():
        # Đóng băng text_encoder và prompt_learner (không train)
        if "text_encoder" in key:
            value.requires_grad_(False)
            continue   
        if "prompt_learner" in key:
            value.requires_grad_(False)
            continue
        # Bỏ qua các head text riêng nếu có
        if "classifier_id_bge" in key:
            continue
        # Nếu parameter đã bị đóng băng từ trước thì bỏ qua
        if not value.requires_grad:
            continue

        # LR mặc định từ config STAGE2
        lr = cfg.SOLVER.STAGE2.BASE_LR
        weight_decay = cfg.SOLVER.STAGE2.WEIGHT_DECAY

        # Xử lý bias riêng
        if "bias" in key:
            lr = cfg.SOLVER.STAGE2.BASE_LR * cfg.SOLVER.STAGE2.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.STAGE2.WEIGHT_DECAY_BIAS

        # Nếu bật LARGE_FC_LR, tăng LR cho các head classifier
        if cfg.SOLVER.STAGE2.get("LARGE_FC_LR", False):
            if "classifier" in key or "arcface" in key:
                lr = cfg.SOLVER.STAGE2.BASE_LR * 2
                print('Using two times learning rate for fc')

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        keys += [key]

    print("Stage2 trainable params:")
    for key in keys:
        print("  ", key)

    # Tạo optimizer theo tên được cấu hình
    opt_name = cfg.SOLVER.STAGE2.OPTIMIZER_NAME
    if opt_name == 'SGD':
        optimizer = torch.optim.SGD(params, momentum=cfg.SOLVER.STAGE2.MOMENTUM)
    elif opt_name == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.STAGE2.BASE_LR, weight_decay=cfg.SOLVER.STAGE2.WEIGHT_DECAY)
    else:
        optimizer = getattr(torch.optim, opt_name)(params)

    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.STAGE2.CENTER_LR)


    return optimizer, optimizer_center


# giữ tương thích với code cũ
def make_optimizer_2stage(cfg, model, center_criterion):
    return make_optimizer_promptsg(cfg, model, center_criterion)