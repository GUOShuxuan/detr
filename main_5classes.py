# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modify the main.py to the nvdata:
1. build_dataset: get the nvdata for training (start from 50K) and evaluate
2. change the evaluate function to evaluate on nvdata
"""
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler, Subset, ConcatDataset

import datasets
import util.misc as utils
# from datasets import get_coco_api_from_dataset
from sandbox.williamz.detr.engine import train_one_epoch
from sandbox.williamz.detr.models import build_model
from sandbox.williamz.detr.datasets.nvidia_5classes import build_nvdataset, build_nvdataset_large
from sandbox.williamz.detr.eval_dlav_metrics import evaluate

import IPython


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int) #
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='nvdata_5classes')
    # parser.add_argument('--coco_path', type=str)
    parser.add_argument('--dataset_root_sql', type=str, default=None)
    parser.add_argument('--dataset_root_sql_3', type=str, default=None)
    parser.add_argument('--dataset_root_sql_4', type=str, default=None)
    parser.add_argument('--dataset_root_sql_5', type=str, default=None)
    parser.add_argument('--dataset_root_sql_6', type=str, default=None)
    parser.add_argument('--dataset_root_sql_7', type=str, default=None)

    parser.add_argument('--dataset_root_img', type=str, default=None)
    parser.add_argument('--dataset_root_test', type=str, default=None) 
    parser.add_argument('--root_indices', type=str, default=None)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--camera', type=str, default='full')
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    ## for autonet
    parser.add_argument('--auto_checkpoint', type=str, default='sandbox/williamz/detr/res_autonet/final_epoch.checkpoint') #'sandbox/williamz/detr/res_autonet/final_epoch.checkpoint'
    parser.add_argument('--training_spec', type=str, default='sandbox/williamz/detr/res_autonet/autonet_training_spec.yaml')
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # IPython.embed()
    # os.system("sudo chmod -R 777 /home/shuxuang/.cache/")
    model, criterion, postprocessors = build_model(args) # use the same model as detr paper on coco
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    if args.dataset_root_sql == args.dataset_root_img:
        dataset_train = build_nvdataset_large(dataset_root=[args.dataset_root_sql,  args.dataset_root_img],
                                        mode='train', camera=args.camera)
        if args.dataset_root_sql_3 is not None and os.path.isdir(args.dataset_root_sql_3):
            dataset_train_3 = build_nvdataset_large(dataset_root=[args.dataset_root_sql_3,  args.dataset_root_sql_3],
                                        mode='train', camera=args.camera)
            if args.dataset_root_sql_4 is not None and os.path.isdir(args.dataset_root_sql_4):
                dataset_train_4 = build_nvdataset_large(dataset_root=[args.dataset_root_sql_4,  args.dataset_root_sql_4],
                                        mode='train', camera=args.camera)
                if args.dataset_root_sql_5 is not None and os.path.isdir(args.dataset_root_sql_5):
                    dataset_train_5 = build_nvdataset_large(dataset_root=[args.dataset_root_sql_5,  args.dataset_root_sql_5],
                                        mode='train', camera=args.camera)
                    if args.dataset_root_sql_6 is not None and os.path.isdir(args.dataset_root_sql_6):
                        dataset_train_6 = build_nvdataset_large(dataset_root=[args.dataset_root_sql_6,  args.dataset_root_sql_6],
                                        mode='train', camera=args.camera)
                        if args.dataset_root_sql_7 is not None and os.path.isdir(args.dataset_root_sql_7):
                            dataset_train_7 = build_nvdataset_large(dataset_root=[args.dataset_root_sql_7,  args.dataset_root_sql_7],
                                        mode='train', camera=args.camera)
                            dataset_train = [dataset_train, dataset_train_3, dataset_train_4, dataset_train_5, dataset_train_6, dataset_train_7]
                        else:
                            dataset_train = [dataset_train, dataset_train_3, dataset_train_4, dataset_train_5, dataset_train_6]
                    else:
                        dataset_train = [dataset_train, dataset_train_3, dataset_train_4, dataset_train_5]
                else:
                    dataset_train = [dataset_train, dataset_train_3, dataset_train_4]
            else:
                dataset_train = [dataset_train, dataset_train_3]
    else:
        dataset_train = build_nvdataset(dataset_root=[args.dataset_root_sql,  args.dataset_root_img],
                                        mode='train', camera=args.camera)
    # dataset_val = build_nvdataset(dataset_root=[args.dataset_root_test, args.dataset_root_test], 
    #                               mode='test', camera=args.camera)
    if args.root_indices is not None:
        indices_50k =np.load(os.path.join(args.root_indices))
        # indices_50k =np.load(os.path.join(os.environ["HOME"],'datasets/id_1_criterion_Max_SSD_num_labels_50000.npy'))
        # for forward_center: 
        # indices_50k = np.load(os.path.join(os.environ["HOME"],'datasets/indices_fc_50000.npy'))
        dataset_train = Subset(dataset_train, indices_50k)
    # IPython.embed()
    if isinstance(dataset_train, list):
        dataset_train = ConcatDataset(dataset_train)

    print("Train samples: %d"%(len(dataset_train)))
    # IPython.embed()
    # if dataset_train is list:
    #     dataset_train = ConcatDataset(dataset_train)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)


    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 50 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)


        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)


# dazel run //detr/workflows:train_detr -- -e maglev --remote_registry ngc
# dazel run //sandbox/williamz/detr:main_5classes -- --batch_size 1 --num_workers 0 

# run on: CUDA_VISIBLE_DEVICES=1 dazel run //sandbox/williamz/detr:main -- --batch_size 4 --dilation --output_dir /home/shuxuang/experiments/detr_dc5_50k_bs4/

# dazel run //sandbox/williamz/detr/workflows:train_detr -- -e maglev --remote_registry ngc

# dazel run //sandbox/williamz/detr:main_5classes -- --batch_size 2  --camera forward_center --dataset_root_sql /home/shuxuang/datasets/detection-f --dataset_root_img /home/shuxuang/datasets/detection-f --dataset_root_sql_3 /home/shuxuang/datasets/largeset --dataset_root_sql_4 /home/shuxuang/datasets/detection-f-ppl-cyclist