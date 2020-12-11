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
from torch.utils.data import DataLoader, DistributedSampler, Subset

import datasets
import util.misc as utils
# from datasets import get_coco_api_from_dataset
from sandbox.williamz.detr.engine import train_one_epoch
from sandbox.williamz.detr.models import build_model
from sandbox.williamz.detr.datasets.nvidia import build_nvdataset
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
    parser.add_argument('--dataset_file', default='nvdata')
    # parser.add_argument('--coco_path', type=str)
    parser.add_argument('--dataset_root_sql', type=str)
    parser.add_argument('--dataset_root_img', type=str)
    parser.add_argument('--dataset_root_test', type=str) 
    parser.add_argument('--root_indices', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
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
    # IPython.embed()
    os.system("sudo chmod -R 777 /home/shuxuang/.cache/")
    model, criterion, postprocessors = build_model(args) # use the same model as detr paper on coco
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
    #                               weight_decay=args.weight_decay)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    # dataset_train = build_dataset(image_set='train', args=args)
    # dataset_val = build_dataset(image_set='val', args=args)
    # modify the dataset from coco to nvdata
    # home_dir = os.environ["HOME"]
    # # on local
    # dataset_train_ = build_nvdataset(dataset_root=[
    #                                     os.path.join(os.environ["HOME"],'datasets/annotation_sql_nvidia'), 
    #                                     os.path.join(os.environ["HOME"], 'datasets/frames_nvidia')], 
    #                                 mode='train')
    dataset_val = build_nvdataset(dataset_root=[
                                    os.path.join(os.environ["HOME"],'datasets/test'), 
                                    os.path.join(os.environ["HOME"], 'datasets/frames_nvidia')], 
                                  mode='test')
    # indices_50k =np.load(os.path.join(os.environ["HOME"],'datasets/id_1_criterion_Max_SSD_num_labels_50000.npy'))
    # # on maglev
    # dataset_train_ = build_nvdataset(dataset_root=[args.dataset_root_sql,  args.dataset_root_img],
    #                                  mode='train')
    # dataset_val = build_nvdataset(dataset_root=[args.dataset_root_test, args.dataset_root_sql], 
    #                               mode='test')
    # indices_50k =np.load(os.path.join(args.root_indices))

    # dataset_train = Subset(dataset_train_, indices_50k)
    print("Validation samples: %d"%(len(dataset_val)))
    # IPython.embed()


    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    # args.resume = os.path.join(os.environ["HOME"], 'datasets/exps_detr_base/checkpoint0299.pth')
    # args.resume = '/home/shuxuang/datasets/exps_detr_base/checkpoint0299.pth'
    print(args.resume)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            print('Loading model: %s'%args.resume)
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])

    if args.eval:
        evaluate(model, dataset_val, postprocessors, device)
    return model, dataset_val, postprocessors, device

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)


# CUDA_VISIBLE_DEVICES=1 dazel run //sandbox/williamz/detr:train_with_eval -- --eval

# test:
# CUDA_VISIBLE_DEVICES=1 dazel run //sandbox/williamz/detr:eval -- --eval --resume /home/shuxuang/experiments/train_output/checkpoint.pth

# get info:
# maglev workflows get 0cf25940-3f00-5c2c-a8e8-1571e986513b
# maglev volumes mount -n train-outputs  -v 4977fea-0998-4d5b-b557-ff17605f2098 -p /home/shuxuang/experiments/train_output