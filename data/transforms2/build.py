import torchvision.transforms as T

def build_transforms(cfg, is_train=True):
    if is_train:
        transform = T.Compose([
            T.Resize(size=(cfg.INPUT.IMAGE_SIZE, cfg.INPUT.IMAGE_SIZE)),
            T.ToTensor()
        ])
    else:
        transform = T.Compose([
            T.Resize(size=(cfg.INPUT.IMAGE_SIZE, cfg.INPUT.IMAGE_SIZE)),
            T.ToTensor()
        ])

    return transform