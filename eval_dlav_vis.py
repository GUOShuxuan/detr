# Copyright (c) 2020 NVIDIA CORPORATION.  All rights reserved.

"""Evaluation script that uses dlav/metrics/detection implementation of mAP."""

import argparse
import os
import time

from dlav.metrics.detection.data.metrics_database import MetricsDatabase
# from dlav.metrics.detection.process.detection_metrics_wrapper import DetectionMetricsWrapper
from dlav.metrics.detection.report import output as dm_output
import torch
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from sandbox.williamz.detr.datasets.nvidia import (
    NVIDIA_CLASSES,
    NVIDIADetection,
)

from sandbox.williamz.secret_project.types import Detection, Label
from sandbox.williamz.detr.detection_metrics_wrapper import DetectionMetricsWrapper

import IPython

# DATASET_MEAN = (104, 117, 123)
# TODO(@williamz): This should really come from the dataset instance. Too brittle
# otherwise. It also needs to be changed for AV data eventually --> best place to
# encapsulate that is in the dataset.
# IDX_TO_CLASS_NAME = {i: class_name for i, class_name in enumerate(VOC_CLASSES)}
# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933],
          [1.000, 0.000, 0.498], [0.000, 0.600, 0.000], [0.800, 0.000, 0.000]]
IDX_TO_CLASS_NAME = {i: class_name for i, class_name in enumerate(NVIDIA_CLASSES)}

def plot_results(pil_img, prob, labels, boxes, out_img):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    for p, cl, (xmin, ymin, xmax, ymax) in zip(prob, labels, boxes.tolist()):
        cl = int(cl)
        c = COLORS[cl] #* 100
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=2))
        # IPython.embed()
        text = f'{IDX_TO_CLASS_NAME[cl]}: {p:0.2f}'
        ax.text(xmin, ymin, text, fontsize=10,
                bbox=dict(facecolor=c, alpha=0.5)) #'yellow'
    plt.axis('off')
    plt.show()
    # plt.savefig(out_img, bbox_inches='tight', pad_inches=0, format='jpg', dpi=300)
    plt.savefig(out_img, dpi=300)
    
def plot_gts(pil_img, labels, boxes, out_img):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    for cl, (xmin, ymin, xmax, ymax) in zip(labels, boxes.tolist()):
        cl = int(cl)
        # IPython.embed()
        c = COLORS[cl] #* 100
        # print(c)
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=2))
        # cl = p.argmax()
        text = f'{IDX_TO_CLASS_NAME[cl]}'
        ax.text(xmin, ymin, text, fontsize=10,
                bbox=dict(facecolor=c, alpha=0.5))
    plt.axis('off')
    plt.show()
    # plt.savefig(out_img,  bbox_inches='tight', pad_inches=0, format='jpg', dpi=300)
    plt.savefig(out_img, dpi=300, bbox_inches='tight', pad_inches=0.05,)
# plot_results(im, scores, boxes)

def plot_gts_results(pil_img, prob, labels, boxes, gt_labels, gt_boxes, out_img):
    '''
    pil_image: the original images before transform in loading the nvdata
    prob: the predicted prob of the predicted classes
    labels: the predicted labels 
    boxes: the predicted boxes
    gt_labels: ground truth lables
    gt_boxes: ground truth boxes before the target transform
    In my case, I do some modification in the pull_item(i):
    orig_img, image, ori_target, target = dataset.pull_item(i, mode='vis'):
    plot_gts_results(orig_img, predictions[..., 1], predictions[..., 0], predictions[..., 2:6],
                        ori_target[:,-1], ori_target[:, :4], 
                        "/home/shuxuang/experiments/demos/detr_results/hc_08/img_q100_300epochs_hc_%d.jpg"%(i))

    '''
    # plt.figure(figsize=(3200, 1000))
    # plt.rcParams.update({'font.size': 6})
    fig, ((ax1), (ax2)) = plt.subplots(1, 2, figsize=(35,10))
    fig.subplots_adjust(wspace=0.0001)
    ax1.imshow(pil_img)
    # ax = plt.gca()
    for cl, (xmin, ymin, xmax, ymax) in zip(gt_labels, gt_boxes.tolist()):
        cl = int(cl)
        c = COLORS[cl] #* 100
        ax1.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=2))
        # IPython.embed()
        text = f'{IDX_TO_CLASS_NAME[cl]}'
        ax1.text(xmin, ymin, text, fontsize=10,
                bbox=dict(facecolor=c, alpha=0.5)) #'yellow'

    ax2.imshow(pil_img)
    for p, cl, (xmin, ymin, xmax, ymax) in zip(prob, labels, boxes.tolist()):
        cl = int(cl)
        c = COLORS[cl] #* 100
        ax2.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=2))
        # IPython.embed()
        text = f'{IDX_TO_CLASS_NAME[cl]}: {p:0.2f}'
        ax2.text(xmin, ymin, text, fontsize=10,
                bbox=dict(facecolor=c, alpha=0.5)) #'yellow'

    ax1.set_title('gt: %d'%(gt_labels.shape[0]), fontsize=12)
    ax2.set_title('pred: %d'%(labels.size(0)), fontsize=12)
    ax1.axis('off')
    ax2.axis('off')
    plt.show()
    # plt.savefig(out_img,  bbox_inches='tight', pad_inches=0, format='jpg', dpi=300)
    plt.savefig(out_img, dpi=300, bbox_inches='tight', pad_inches=0.2,)


def vis_bboxes(model, dataset, postprocessors, device, out_dir=None):
    """Main function."""
    # The +1 for num_classes is for the background class.
    # This line is needed because the model is typically trained with
    # torch.nn.DataParallel, which prefixes state keys with "module".
    # TODO(@williamz): Write a util that allows loading the state dict
    # regardless of whether the model has been wrapped by DataParallel. 
    model.eval()
  
    # These will be passed to dlav/metrics/detection, and will accumulate the data in
    # a format compatible with that code.
    # frames, labels, detections = [], [], []
    # TODO(@williamz): Switch to batched mode once the `pull_item` has been put
    # into `__getitem__` directly.
    sample_set = [0, 20, 100, 1000, 2000, 3000, 5000, 6000, 10000, 20000, 30000, 40000]
    # print(len(dataset))
    # sample_set = [20, 100, 2000] # 2000, 10000
    # sample_set = [2000]
    time = 0
    for i in sample_set:  # need to implement returning the length of the dataset #len(dataset)

        # print("Processing Image %d out of %d" % (i, len(sample_set)))
        # image is a CHW tensor.
        # target is a [num_objects, 5] numpy.array, where the 5 values are
        # [L, T, R, B, class_index].
        # image, target = dataset.pull_item(i)
        # IPython.embed()
        orig_img, image, ori_target, target = dataset.pull_item(i, mode='vis') # orig_img rgb
        IPython.embed()
        
        # Add batch dimension (unfortunately singular now...).
        image = image.to(device)
        # target = target.to(device)

        # predictions is a [num_classes, max_detections, 5] tensor.
        # The last dimension represents [score, left, top, right, bottom].
        # predictions = model(image).squeeze(dim=0)
        outputs = model(image.unsqueeze(0))
        # outputs['pred_boxes'], outputs['pred_logits']
        # orig_target_sizes = target["orig_size"]
        # results = postprocessors['bbox'](outputs, orig_target_sizes) # [tensor([604, 960])]
        orig_target_sizes = torch.stack([t["orig_size"] for t in [target]], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes.to(device)) 
        #  results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        # ? what are they in results?
        # Translate to image space coordinates.
        # len_boxes = target['boxes'].size(0)
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            len_prediction = model.module.num_queries 
        else:
            len_prediction = model.num_queries 
        predictions = torch.zeros((len_prediction, 6))
   
        predictions[..., 2:6] = results[0]['boxes']
        # # _, pred_cls = results[0]['scores'].max(1)
        predictions[..., 1] = results[0]['scores']
        predictions[..., 0] = results[0]['labels']

        # add a threshold to keep predictions with high confidence
        score_threshold = 0.8
        mask = predictions[..., 1] > score_threshold
        predictions = predictions[mask]
        
        # predictions  len_prediction*[labels, score, box]
        print("ImageID: %d, #GT_boxes: %d,  #Pred_boxes: %d"%(i, ori_target.shape[0], predictions.size(0)))
        # plot_gts(orig_img, ori_target[:,-1], ori_target[:, :4],
        #         "/home/shuxuang/experiments/demos/detr_results/img_gt_%d_n%d.jpg"%(i, ori_target.shape[0]))
        # #prob, labels, boxes, out_img
        # plot_results(orig_img, predictions[..., 1], predictions[..., 0], predictions[..., 2:6],
        #             "/home/shuxuang/experiments/demos/detr_results/img_pred_q100_%d_n%d.jpg"%(i, predictions.size(0)))

        plot_gts_results(orig_img, predictions[..., 1], predictions[..., 0], predictions[..., 2:6],
                        ori_target[:,-1], ori_target[:, :4], 
                        "/home/shuxuang/experiments/demos/detr_results/hc_08/forward_center/img_q100_300epochs_rgb_hc_%d.jpg"%(i))
        # IPython.embed()
        # compute mAP for each image
        frames, labels, detections = [], [], []
        frames.append(str(i))
        labels += _get_frame_labels(i, ori_target) #gt_truth is correct with (x1, y1, x2, y2)
        
        # Skip class_index = num_classes, since it's no object according to the training data.
        start_det_idx = 0
        for class_index in range(0, 11-1): 
            # class prediction is a [max_detections, 5] tensor.
            cls_mask = predictions[...,0] == class_index
            class_predictions = predictions[cls_mask][..., 1:6]
            
            ## Is this a must-do?
            # class_predictions, _ = class_predictions.sort(0, descending=True) # decending, keep the predictions score from large to small
            # fixed on 12.10.2020
            if class_predictions.size(0)>0:
                class_predictions  = torch.stack(sorted(class_predictions, key=lambda class_predictions: -class_predictions[0]))
            # First, keep only predictions whose score is strictly greater than 0.0.
            # IPython.embed()
            # ious = box_iou(class_predictions[:,1:5], torch.from_numpy(ori_target[:,0:4]).float())
            # mask is a [max_detections] tensor.
            mask = class_predictions[:, 0].gt(0.0)
            class_predictions = torch.masked_select(
                    class_predictions, mask.unsqueeze(-1)
                ).view(-1, 5).cpu().numpy()
            current_detections = _get_frame_detections(
                i, class_index, class_predictions, start_det_idx
            )
            start_det_idx += len(current_detections)
            detections += current_detections
            # IPython.embed()
        # ious = box_iou(predictions[..., 2:6], torch.Tensor(ori_target[:, :4]))
        # (ious>0.5).sum(1)
        # iou = box_iou(torch.Tensor(detections[17][2].bbox).view(-1, 4), torch.Tensor(labels[1][2].bbox).view(-1, 4))
        # iou of (pred, gt)
        ## 2000
        # person: 16, 22: 0.5030
        # traffic light: 19-0:0., 2: 0.5958, 4:0, 5:0,  6: 0.1231 
        # road_sign: 17-1/3: 0
        with MetricsDatabase.create(path=":memory:", mode="w") as db_out:
            db_out.export_images_iter(iterator=frames, ignore_duplicates=True)
            db_out.export_detections_iter(iterator=detections, ignore_duplicates=True)
            db_out.export_groundtruths_iter(iterator=labels, ignore_duplicates=True)

            metrics_wrapper = DetectionMetricsWrapper(
                database=db_out,
                # TODO(@williamz): allow this to be passed in.
                #configuration=evaluation_settings,
            )

            results = metrics_wrapper.results()

            # TODO(@williamz): investigate delegating all printing to metrics code (with possible
            # changes upstream).
            dm_output.print_average_precision(results)

def inference_time(model, dataset, postprocessors, device, out_dir=None):
    """Main function."""

    model.eval()
    _t = {'im_detect': Timer(), 'misc': Timer()}
    time = 0
    for i in range(1000):  # need to implement returning the length of the dataset #len(dataset)

        orig_img, image, ori_target, target = dataset.pull_item(i, mode='vis') # orig_img rgb
       
        image = image.to(device).unsqueeze(0)
   
        _t['im_detect'].tic()
        outputs = model(image)
        detect_time = _t['im_detect'].toc(average=False)
        time = time + detect_time
        print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1,
                                                    1000, detect_time))
    print(time/1000)


def _get_frame_labels(frame_id, target):
    """Get labels for a given a frame.

    Args:
        frame_id (int): Frame index.
        target (numpy.array): [num_objects, 5] array where the 5 values are
            [L, T, R, B, class_index].

    Returns:
        List of (frame_id, label_id, Label) triplets.
    """
    num_labels = target.shape[0]
    frame_labels = []
    for i in range(num_labels):
        frame_labels.append(
            (
                str(frame_id),
                i,
                Label(
                    class_name=IDX_TO_CLASS_NAME[int(target[i, -1])],
                    bbox=target[i, :4].tolist(),
                ),
            )

        )

    return frame_labels


def _get_frame_detections(frame_id, class_index, class_predictions, start_det_idx):
    """Get detections for a given frame.

    Args:
        frame_id (int): Frame index.
        class_index (int): Class index.
        class_predictions (numpy.array): [num_detections, 5] array where the 5
            values are [score, L, T, R, B].
        start_det_idx (int): Index offset from which to start indexing additional
            detections.

    Returns:
        List of (frame_id, detection_id, Detection) triplets.
    """
    num_predictions = class_predictions.shape[0]
    frame_detections = []
    for i in range(num_predictions):
        frame_detections.append(
            (
                str(frame_id),
                start_det_idx + i,  # HACK.
                Detection(
                    # -1 because IDX_TO_CLASS_NAME doesn't have the background class.
                    # class_name=IDX_TO_CLASS_NAME[class_index - 1],
                    class_name=IDX_TO_CLASS_NAME[class_index],
                    bbox=class_predictions[i, 1:5].tolist(),
                    confidence=class_predictions[i, 0],
                ),
            )
        )

    return frame_detections


# compute iou 
def box_area(boxes):
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.

    Arguments:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format

    Returns:
        area (Tensor[N]): area for each box
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def box_iou(boxes1, boxes2):
    """
    Return intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.

    Arguments:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])

    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou


class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

