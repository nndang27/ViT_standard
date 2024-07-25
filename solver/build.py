import torch


def make_optimizer(cfg, model):
    optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(
                             params=model.parameters(), 
                             lr=cfg.SOLVER.BASE_LR,
                             betas=(0.9, 0.999), 
                             weight_decay=cfg.SOLVER.WEIGHT_DECAY_BIAS
                             )
    return optimizer