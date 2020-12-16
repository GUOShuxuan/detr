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
from sandbox.williamz.detr.eval_dlav_vis import vis_bboxes, inference_time
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

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
    parser.add_argument('--camera', type=str, default='full')

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

def read_log(logfile):
    loss_dict = {}
    with open(logfile) as f:
        lines = f.readlines()
    nline = 0
    for line in lines:
        line_dict = json.loads(line)
        for key in line_dict.keys():
            if nline == 0:
                loss_dict.update({key: []})
            loss_dict[key].append(line_dict[key])
        nline += 1
    # IPython.embed()
    plt.figure(figsize=(5, 10))
    plt.rcParams.update({'font.size': 6})
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
    fig.suptitle('DETR Baseline with 100 queries', fontsize=10)
    fig.subplots_adjust(top=0.90, wspace=0.25, hspace=0.25) # top=0.7
    # fig, axes = plt.subplots(nrows=2, ncols=2)
    # ax0, ax1, ax2, ax3 = axes.flatten()
    for key in loss_dict.keys():
        if key.find('unscaled') == -1:
            if key.find('class_error') != -1:
                ax1.plot(loss_dict["epoch"], loss_dict[key])
            elif key.find('loss_bbox') != -1:
                ax4.plot(loss_dict["epoch"], loss_dict[key], label=key)
            elif key.find('loss_ce') != -1:
                ax5.plot(loss_dict["epoch"], loss_dict[key], label=key)
            elif key.find('loss_giou') != -1:
                ax6.plot(loss_dict["epoch"], loss_dict[key], label=key)
            elif key.find('loss') != -1:
                ax2.plot(loss_dict["epoch"], loss_dict[key])
        elif key.find('cardinality_error') != -1:
            ax3.plot(loss_dict["epoch"], loss_dict[key], label=key)
        
    ax1.set_title('class error', fontsize=8)
    ax2.set_title('loss', fontsize=8)
    ax3.set_title('cardinality_error', fontsize=8)
    ax4.set_title('loss_bbox', fontsize=8)
    ax5.set_title('loss_ce', fontsize=8)
    ax6.set_title('loss_giou', fontsize=8)
    ax3.legend(prop={'size': 6})
    ax4.legend(prop={'size': 6})
    ax5.legend(prop={'size': 6})
    ax6.legend(prop={'size': 6})
    
    # fig.tight_layout()
    plt.show()
    plt.savefig("/home/shuxuang/experiments/demos/detr_losses/baseline_rgb_q100_300epochs_front_camera.png", dpi=300)

    # epoch-train-class-error, epoch-train-loss
    # epoch-train-loss_boxes, ce, giou * 5, train_cardinality_error*5
    

def accumulate_bboxes_numbers(dataset):
    """Main function."""
    # The +1 for num_classes is for the background class.
    # This line is needed because the model is typically trained with
    # torch.nn.DataParallel, which prefixes state keys with "module".
    # TODO(@williamz): Write a util that allows loading the state dict
    # regardless of whether the model has been wrapped by DataParallel. 
    # n_50 = []
    # n_100 = []
    # n_150 = []
    # n_200 = []
    # n_larger_200 = []

    ns_list = [[], [], [], [], []]  
    # ns_list[3]: [38223, 38224]
    # ns_list[2]: [36614, 36615, 38177, 38178, 38221, 38222, 38225, 38323, 38325, 39362, 43032, 47159, 47160, 47161, 47162, 47163, 47165]
    # ns_list[1][::200]: [524, 5591, 9440, 11904, 13810, 16593, 38193, 42916, 43602, 45248, 46297, 47572, 48546, 49084]
    # ns_list[0][::4000]: [0, 4127, 8296, 12651, 17021, 21053, 25069, 29096, 33117, 37190, 41259, 45948]
    ns = [0, 0, 0, 0, 0] # test: [46988, 2777, 17, 2, 0]

    for i in range(len(dataset)):  # need to implement returning the length of the dataset #len(dataset)

        _, _, ori_target, _ = dataset.pull_item_vis(i)
        # IPython.embed()
        if i%1000 == 0:
            print("Processing Image %d out of %d" % (i, len(dataset)))
        index = ori_target.shape[0] // 50
        ns_list[index].append(i) 
        ns[index] += 1
    
    # IPython.embed()  
    print(ns)
    with open('/home/shuxuang/experiments/demos/data/test/n_objects.json', 'w') as f:
        json.dump(ns_list, f)


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


    # dataset_val = build_nvdataset(dataset_root=[
    #                                 os.path.join(os.environ["HOME"],'datasets/test'), 
    #                                 os.path.join(os.environ["HOME"], 'datasets/frames_nvidia')], 
    #                               mode='test', camera=args.camera)
    dataset_val = build_nvdataset(dataset_root=[args.dataset_root_test, args.dataset_root_sql], 
                                  mode='test', camera=args.camera)

    print("Validation samples: %d"%(len(dataset_val)))
    # IPython.embed()
    # compute how many boxes in the test dataset for each image
    # accumulate_bboxes_numbers(dataset_val)
    # dataset_train_ = build_nvdataset(dataset_root=[
    #                                     os.path.join(os.environ["HOME"],'datasets/annotation_sql_nvidia'), 
    #                                     os.path.join(os.environ["HOME"], 'datasets/frames_nvidia')], 
    #                                 mode='train')

    # indices_50k =np.load(os.path.join(os.environ["HOME"],'datasets/id_1_criterion_Max_SSD_num_labels_50000.npy'))
    # dataset_train = Subset(dataset_train_, indices_50k)
    # print(len(dataset_train_))
    # print(len(dataset_val))


    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    # args.resume = os.path.join(os.environ["HOME"], 'datasets/exps_detr_base/checkpoint0299.pth')
    # args.resume = '/home/shuxuang/datasets/exps_detr_base/checkpoint0299.pth'
    log_path = args.resume
    log = os.path.join(args.resume, 'log.txt')
    # read_log(log) 
    # IPython.embed()
    args.resume = os.path.join(args.resume, 'checkpoint.pth')
    print(args.resume)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            print('Loading model: %s'%args.resume)
            checkpoint = torch.load(args.resume, map_location='cpu')
        print('Load model from %d epoch' % checkpoint['epoch'])
        model_without_ddp.load_state_dict(checkpoint['model'])

    if args.eval:
        # vis_bboxes(model, dataset_val, postprocessors, device)
        inference_time(model, dataset_val, postprocessors, device)
    return model, dataset_val, postprocessors, device

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)


# CUDA_VISIBLE_DEVICES=1 dazel run //sandbox/williamz/detr:eval_vis -- --eval --resume /home/shuxuang/experiments/detr_results/20201207/baseline_100

# CUDA_VISIBLE_DEVICES=1 dazel run //sandbox/williamz/detr:eval_vis -- --eval --resume /home/shuxuang/experiments/detr_results/20201207/baseline_100
# If change the resume path to the model, then change 
#  - the title of the loss figure, 
#  - the name of the saved loss figure file 
#  - the name of samples in eval_dlav_vis.

# CUDA_VISIBLE_D0VICES=0 dazel run //sandbox/williamz/detr:eval_vis -- --eval --resume /home/shuxuang/experiments/detr_results/20201207/baseline_100/ --num_queries 100 --camera full