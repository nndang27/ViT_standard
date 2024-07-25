from config import cfg
from utils.logger import setup_logger
import argparse
from utils import EarlyStopping
import torch
from engine import train
from solver import make_optimizer
from modeling.vit_model import ViT
from data import dowload_animal_datasets, build_datasets, build_dataloader

def set_seed(seed:int=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def main():
    parser = argparse.ArgumentParser(description="PyTorch Template MNIST Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file != '':
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    
    set_seed()

    train_path, test_path = dowload_animal_datasets()
    print("Train_PATH: ", train_path)
    print("Test_PATH: ", test_path)
    train_dataset = build_datasets(cfg, train_path)
    test_dataset = build_datasets(cfg, test_path)
    train_dataloaders = build_dataloader(cfg, train_dataset, is_train=True)
    test_dataloader = build_dataloader(cfg, test_dataset, is_train=False)

    model = ViT('B_16_imagenet1k', pretrained=True, image_size=224)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = make_optimizer(cfg, model)
    early_stopping = EarlyStopping(mode='min', patience=10)

    devices = "cuda" if torch.cuda.is_available() else "cpu"
    model_result = train(model=model,train_dataloader=train_dataloaders,test_dataloader=test_dataloader,optimizer=optimizer,loss_fn=loss_fn,epochs=60,early_stopping=early_stopping, devices=devices)

if __name__ == '__main__':
    main()