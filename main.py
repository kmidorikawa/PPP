import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils.common_utils import create_logger
from pcdet.utils.train_utils import train_model
import argparse
import os

def parse_config():
    parser = argparse.ArgumentParser(description='Train PointPillar on Pedestrian class')
    parser.add_argument('--cfg_file', type=str, default='configs/pointpillar_kitti_pedestrian.yaml',
                        help='Path to the training config file')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=80, help='Number of epochs')
    args = parser.parse_args()
    return args

def main():
    args = parse_config()
    cfg_from_yaml_file(args.cfg_file, cfg)
    logger = create_logger()
    
    # Dataloader
    train_loader, val_loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=['Pedestrian'],
        batch_size=args.batch_size,
        dist=False, workers=4, training=True
    )
    
    # Model
    model = build_network(model_cfg=cfg.MODEL, num_class=1, dataset=train_loader.dataset)
    model.cuda()
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.OPTIMIZATION.LR, weight_decay=cfg.OPTIMIZATION.WEIGHT_DECAY)
    
    train_model(model, train_loader, val_loader, optimizer, args.epochs, logger)
    
if __name__ == '__main__':
    main()
