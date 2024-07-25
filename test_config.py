from config import cfg
from utils.logger import setup_logger
import argparse

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
    
    output_dir = "E:/ViT_folder/ViT_standard"
    logger = setup_logger("template_model", output_dir, 0)
    logger.info("Running with config:\n{}".format(cfg))
    return

if __name__ == '__main__':
    main()
