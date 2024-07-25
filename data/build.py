from .transforms2 import build_transforms
from torchvision import datasets
from torch.utils.data import DataLoader

def build_datasets(cfg, folder_dir):
    transform = build_transforms(cfg, is_train=True)
    dataset = datasets.ImageFolder(folder_dir,transform=transform)
    return dataset

def build_dataloader(cfg, dataset, is_train):
    if is_train:
        batch_size = cfg.SOLVER.IMS_PER_BATCH
        shuffle = True
    else:
        batch_size = cfg.TEST.IMS_PER_BATCH
        shuffle = False

    num_workers = cfg.DATALOADER.NUM_WORKERS
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    return dataloader