# Copyright (c) 2020 NVIDIA CORPORATION.  All rights reserved.

"""Evaluation script that uses dlav/metrics/detection implementation of mAP."""

import argparse
import os
import time

from dlav.metrics.detection.data.metrics_database import MetricsDatabase
# from dlav.metrics.detection.process.detection_metrics_wrapper import DetectionMetricsWrapper
from dlav.metrics.detection.report import output as dm_output
import torch
import numpy as np
# import matplotlib
# from matplotlib import pyplot as plt

from sandbox.williamz.detr.datasets.nvidia import (
    NVIDIA_CLASSES,
    NVIDIADetection,
)

from sandbox.williamz.secret_project.types import Detection, Label
from sandbox.williamz.detr.detection_metrics_wrapper import DetectionMetricsWrapper

import IPython

DATASET_MEAN = (104, 117, 123)
# TODO(@williamz): This should really come from the dataset instance. Too brittle
# otherwise. It also needs to be changed for AV data eventually --> best place to
# encapsulate that is in the dataset.
# IDX_TO_CLASS_NAME = {i: class_name for i, class_name in enumerate(VOC_CLASSES)}
# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
IDX_TO_CLASS_NAME = {i: class_name for i, class_name in enumerate(NVIDIA_CLASSES)}

# def plot_results(pil_img, prob, boxes):
#     plt.figure(figsize=(16,10))
#     plt.imshow(pil_img)
#     ax = plt.gca()
#     for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), COLORS * 100):
#         ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
#                                    fill=False, color=c, linewidth=3))
#         cl = p.argmax()
#         text = f'{IDX_TO_CLASS_NAME[cl]}: {p[cl]:0.2f}'
#         ax.text(xmin, ymin, text, fontsize=15,
#                 bbox=dict(facecolor='yellow', alpha=0.5))
#     plt.axis('off')
#     plt.show()
    
# plot_results(im, scores, boxes)

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

def evaluate_mAP100(model, dataset, postprocessors, device, out_dir=None):
    "comments: by this and PostProcess_mAP100 in model/detr, wecan get 100 mAP for each class"
    """Main function."""
    # The +1 for num_classes is for the background class.
    # This line is needed because the model is typically trained with
    # torch.nn.DataParallel, which prefixes state keys with "module".
    # TODO(@williamz): Write a util that allows loading the state dict
    # regardless of whether the model has been wrapped by DataParallel. 
    model.eval()
  
    # These will be passed to dlav/metrics/detection, and will accumulate the data in
    # a format compatible with that code.
    frames, labels, detections = [], [], []
    # TODO(@williamz): Switch to batched mode once the `pull_item` has been put
    # into `__getitem__` directly.
    for i in range(2000):  # need to implement returning the length of the dataset #len(dataset)
        # for i in range(len(dataset)):
        if i % 1000 == 0:
            # print("Image %d out of %d" % (i, len(dataset)))
            print("Image %d out of %d" % (i, len(dataset)))
        # image is a CHW tensor.
        # target is a [num_objects, 5] numpy.array, where the 5 values are
        # [L, T, R, B, class_index].
        # image, target = dataset.pull_item(i)
        # IPython.embed()
        image, ori_target, target = dataset.pull_item_eval(i)
        # Add batch dimension (unfortunately singular now...).
        image = image.to(device)
        # target = target.to(device)

        # predictions is a [num_classes, max_detections, 5] tensor.
        # The last dimension represents [score, left, top, right, bottom].
        # predictions = model(image).squeeze(dim=0)
        # outputs = model(image.unsqueeze(0))
        # outputs['pred_boxes'], outputs['pred_logits']
        # orig_target_sizes = target["orig_size"]
        # results = postprocessors['bbox'](outputs, orig_target_sizes) # [tensor([604, 960])]
        orig_target_sizes = torch.stack([t["orig_size"] for t in [target]], dim=0)
        results = postprocessors['bbox'](target, orig_target_sizes) 
        #  results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        # ? what are they in results?
        # Translate to image space coordinates.
        # len_boxes = target['boxes'].size(0)
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            len_prediction = model.module.num_queries 
        else:
            len_prediction = model.num_queries 
        num_classes=11
        len_prediction = ori_target.shape[0]
        predictions = torch.zeros((len_prediction, 6))
        # # predictions = 
        # for i in range(num_classes-1):

        predictions[..., 2:6] = results[0]['boxes']
        # # _, pred_cls = results[0]['scores'].max(1)
        predictions[..., 1] = results[0]['scores']
        predictions[..., 0] = results[0]['labels']
        # predictions  len_prediction*[labels, score, box]
        # get rid of some predictions with quite small scores
        # IPython.embed()
        # score_threshold = 0.9 # 0.01
        # mask = predictions[..., 1] > score_threshold
        # # IPython.embed()
        # predictions = predictions[mask]

        frames.append(str(i))
        labels += _get_frame_labels(i, ori_target) #gt_truth is correct with (x1, y1, x2, y2)
        
        # Skip class_index = num_classes, since it's no object according to the training data.
        start_det_idx = 0
        for class_index in range(0, num_classes-1): 
            # class prediction is a [max_detections, 5] tensor.
            cls_mask = predictions[...,0] == class_index
            class_predictions = predictions[cls_mask][..., 1:6]  
            ## Is this a must-do?
            # class_predictions_, _ = class_predictions.sort(0, descending=True) # decending, keep the predictions score from large to small
            # if class
            # IPython.embed()
            if class_predictions.size(0)>0:
                class_predictions  = torch.stack(sorted(class_predictions, key=lambda class_predictions: class_predictions[0]))

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
            # IPython.embed()
            start_det_idx += len(current_detections)
            detections += current_detections
        # IPython.embed()
    # IPython.embed()
    # with MetricsDatabase.create(path=":memory", mode="w") as db_out:
    # with MetricsDatabase.create(path=os.path.join(out_dir, "memory"), mode="w") as db_out:
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


def evaluate(model, dataset, postprocessors, device, out_dir=None):
    """Main function."""
    # The +1 for num_classes is for the background class.
    # This line is needed because the model is typically trained with
    # torch.nn.DataParallel, which prefixes state keys with "module".
    # TODO(@williamz): Write a util that allows loading the state dict
    # regardless of whether the model has been wrapped by DataParallel. 
    model.eval()
  
    # These will be passed to dlav/metrics/detection, and will accumulate the data in
    # a format compatible with that code.
    frames, labels, detections = [], [], []
    # TODO(@williamz): Switch to batched mode once the `pull_item` has been put
    # into `__getitem__` directly.
    for i in range(len(dataset)):  # need to implement returning the length of the dataset #len(dataset)
        # for i in range(len(dataset)):
        if i % 1000 == 0:
            # print("Image %d out of %d" % (i, len(dataset)))
            print("Image %d out of %d" % (i, len(dataset)))
        # image is a CHW tensor.
        # target is a [num_objects, 5] numpy.array, where the 5 values are
        # [L, T, R, B, class_index].
        # image, target = dataset.pull_item(i)
        # IPython.embed()
        image, ori_target, target = dataset.pull_item(i, mode='test')
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
        num_classes=11
        predictions = torch.zeros((len_prediction, 6))
        # # predictions = 
        # for i in range(num_classes-1):

        predictions[..., 2:6] = results[0]['boxes']
        # # _, pred_cls = results[0]['scores'].max(1)
        predictions[..., 1] = results[0]['scores']
        predictions[..., 0] = results[0]['labels']
        # predictions  len_prediction*[labels, score, box]
        # get rid of some predictions with quite small scores
        # score_threshold = 0.01 # 0.01
        # mask = predictions[..., 1] > score_threshold
        # # # IPython.embed()
        # predictions = predictions[mask]

        frames.append(str(i))
        labels += _get_frame_labels(i, ori_target) #gt_truth is correct with (x1, y1, x2, y2)
        
        # Skip class_index = num_classes, since it's no object according to the training data.
        start_det_idx = 0
        for class_index in range(0, num_classes-1): 
            # class prediction is a [max_detections, 5] tensor.
            cls_mask = predictions[...,0] == class_index
            class_predictions = predictions[cls_mask][..., 1:6]
            
            ## Is this a must-do?
            ## | A bug, with this line, it decending all demensions 
            # class_predictions, _ = class_predictions.sort(0, descending=True) # decending, keep the predictions score from large to small
            # fixed on 12.10.2020
            if class_predictions.size(0)>0:
                class_predictions  = torch.stack(sorted(class_predictions, key=lambda class_predictions: -class_predictions[0])) # decending order according to conf,larger to smaller
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
    # IPython.embed()
    # with MetricsDatabase.create(path=":memory", mode="w") as db_out:
    # with MetricsDatabase.create(path=os.path.join(out_dir, "memory"), mode="w") as db_out:
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

def evaluate_5classes_v0(model, dataset, postprocessors, device, out_dir=None):
    """Main function."""
    ## 
    model.eval()
  
    # These will be passed to dlav/metrics/detection, and will accumulate the data in
    # a format compatible with that code.
    frames, labels, detections = [], [], []
    labels_huge, labels_tiny, labels_small, labels_medium = [], [], [], []
    # TODO(@williamz): Switch to batched mode once the `pull_item` has been put
    # into `__getitem__` directly.
    for i in range(200):  # need to implement returning the length of the dataset #len(dataset)
        # for i in range(len(dataset)):
        if i % 1000 == 0:
            # print("Image %d out of %d" % (i, len(dataset)))
            print("Image %d out of %d" % (i, len(dataset)))
       
        image, ori_target, target = dataset.pull_item(i, mode='test')
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
        num_classes=6 # 5+1
        predictions = torch.zeros((len_prediction, 6))
       
        predictions[..., 2:6] = results[0]['boxes']
        # # _, pred_cls = results[0]['scores'].max(1)
        predictions[..., 1] = results[0]['scores']
        predictions[..., 0] = results[0]['labels']
  

        frames.append(str(i))
        labels += _get_frame_labels(i, ori_target) #gt_truth is correct with (x1, y1, x2, y2)

        # from gt set tiny, small, medium, huge
        # frams for tiny, small, medium, huge:
                ## 1. height?
        # h = ori_target[..., 3] - ori_target[..., 1]
        # # IPython.embed()
        # mask_huge = h > 100.
        # mask_tiny = h <= 8.
        # mask_small = np.bitwise_and(h > 8., h <= 25) # np.bitwise_and(
        # mask_medium = np.bitwise_and(h > 25., h <=100.)

        # # IPython.embed()
        # labels_huge += _get_frame_labels(i, ori_target[mask_huge]) # 4
        # labels_tiny += _get_frame_labels(i, ori_target[mask_tiny]) # 7
        # labels_small += _get_frame_labels(i, ori_target[mask_small]) # 16
        # labels_medium += _get_frame_labels(i, ori_target[mask_medium]) # 12
        
        # Skip class_index = num_classes, since it's no object according to the training data.
        start_det_idx = 0
        for class_index in range(0, num_classes-1): 
            # class prediction is a [max_detections, 5] tensor.
            cls_mask = predictions[...,0] == class_index
            class_predictions = predictions[cls_mask][..., 1:6]
            
            ## Is this a must-do?
            ## | A bug, with this line, it decending all demensions 
            # class_predictions, _ = class_predictions.sort(0, descending=True) # decending, keep the predictions score from large to small
            # fixed on 12.10.2020
            if class_predictions.size(0)>0:
                class_predictions  = torch.stack(sorted(class_predictions, key=lambda class_predictions: -class_predictions[0])) # decending order according to conf,larger to smaller
            
            mask = class_predictions[:, 0].gt(0.0)
            class_predictions = torch.masked_select(
                    class_predictions, mask.unsqueeze(-1)
                ).view(-1, 5).cpu().numpy()
            current_detections = _get_frame_detections(
                i, class_index, class_predictions, start_det_idx
            )

            start_det_idx += len(current_detections)
            detections += current_detections

     
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
    print('---total')
    dm_output.print_average_precision(results)
    print('---huge')
    

def evaluate_5classes(model, dataset, postprocessors, device, out_dir=None):
    """Main function."""
    # The +1 for num_classes is for the background class.
    # This line is needed because the model is typically trained with
    # torch.nn.DataParallel, which prefixes state keys with "module".
    # TODO(@williamz): Write a util that allows loading the state dict
    # regardless of whether the model has been wrapped by DataParallel. 
    model.eval()
  
    # These will be passed to dlav/metrics/detection, and will accumulate the data in
    # a format compatible with that code.
    frames, labels, detections = [], [], []
    labels_huge, labels_tiny, labels_small, labels_medium = [], [], [], []
    detections_huge, detections_tiny, detections_small, detections_medium = [], [], [], []
    # TODO(@williamz): Switch to batched mode once the `pull_item` has been put
    # into `__getitem__` directly.
    for i in range(len(dataset)):  # need to implement returning the length of the dataset #len(dataset)
        # for i in range(len(dataset)):
        if i % 1000 == 0:
            # print("Image %d out of %d" % (i, len(dataset)))
            print("Image %d out of %d" % (i, len(dataset)))
       
        image, ori_target, target = dataset.pull_item(i, mode='test')
        # Add batch dimension (unfortunately singular now...).
        image = image.to(device)
       
        outputs = model(image.unsqueeze(0))

        orig_target_sizes = torch.stack([t["orig_size"] for t in [target]], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes.to(device)) 

        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            len_prediction = model.module.num_queries 
        else:
            len_prediction = model.num_queries 
        num_classes=6
        predictions = torch.zeros((len_prediction, 6))
       
        predictions[..., 2:6] = results[0]['boxes']
        # # _, pred_cls = results[0]['scores'].max(1)
        predictions[..., 1] = results[0]['scores']
        predictions[..., 0] = results[0]['labels']
  

        frames.append(str(i))
        labels += _get_frame_labels(i, ori_target) #gt_truth is correct with (x1, y1, x2, y2)

        # from gt set tiny, small, medium, huge
        # frams for tiny, small, medium, huge:
                ## 1. height?
        h = ori_target[..., 3] - ori_target[..., 1]
        # IPython.embed()
        mask_huge = h > 100.
        mask_tiny = h <= 8.
        mask_small = np.bitwise_and(h > 8., h <= 25) # np.bitwise_and(
        mask_medium = np.bitwise_and(h > 25., h <=100.)

        # IPython.embed()
        labels_huge += _get_frame_labels(i, ori_target[mask_huge]) # 4
        labels_tiny += _get_frame_labels(i, ori_target[mask_tiny]) # 7
        labels_small += _get_frame_labels(i, ori_target[mask_small]) # 16
        labels_medium += _get_frame_labels(i, ori_target[mask_medium]) # 12
        
        # Skip class_index = num_classes, since it's no object according to the training data.
        start_det_idx = 0
        start_det_idx_huge, start_det_idx_medium, start_det_idx_small, start_det_idx_tiny = 0., 0., 0., 0.

        for class_index in range(0, num_classes-1): 
            # class prediction is a [max_detections, 5] tensor.
            cls_mask = predictions[...,0] == class_index
            class_predictions = predictions[cls_mask][..., 1:6]
            
            ## Is this a must-do?
            ## | A bug, with this line, it decending all demensions 
            # class_predictions, _ = class_predictions.sort(0, descending=True) # decending, keep the predictions score from large to small
            # fixed on 12.10.2020
            if class_predictions.size(0)>0:
                class_predictions  = torch.stack(sorted(class_predictions, key=lambda class_predictions: -class_predictions[0])) # decending order according to conf,larger to smaller
            
            
            h_ = class_predictions[..., 4] - class_predictions[..., 2]
            # IPython.embed()
            mask_huge_ = h_ > 100.
            mask_tiny_ = h_ <= 8.
            mask_small_ = torch.bitwise_and(h_ > 8., h_ <= 25) # np.bitwise_and(
            mask_medium_ = torch.bitwise_and(h_ > 25., h_ <=100.)
            # ---huge
            class_predictions_huge = torch.masked_select(
                    class_predictions, mask_huge_.unsqueeze(-1)
                ).view(-1, 5).cpu().numpy()
            current_detections_huge = _get_frame_detections(
                i, class_index, class_predictions_huge, start_det_idx_huge
            )
            
            start_det_idx_huge += len(current_detections_huge)
            detections_huge += current_detections_huge
            ## ---
            # ---medium
            class_predictions_medium = torch.masked_select(
                    class_predictions, mask_medium_.unsqueeze(-1)
                ).view(-1, 5).cpu().numpy()
            current_detections_medium = _get_frame_detections(
                i, class_index, class_predictions_medium, start_det_idx_medium
            )

            start_det_idx_medium += len(current_detections_medium)
            detections_medium += current_detections_medium
            ## ---
            # ---small
            class_predictions_small = torch.masked_select(
                    class_predictions, mask_small_.unsqueeze(-1)
                ).view(-1, 5).cpu().numpy()
            current_detections_small = _get_frame_detections(
                i, class_index, class_predictions_small, start_det_idx_small
            )

            start_det_idx_small += len(current_detections_small)
            detections_small += current_detections_small
            ## ---
            # ---tiny
            class_predictions_tiny = torch.masked_select(
                    class_predictions, mask_tiny_.unsqueeze(-1)
                ).view(-1, 5).cpu().numpy()
            current_detections_tiny = _get_frame_detections(
                i, class_index, class_predictions_tiny, start_det_idx_tiny
            )

            start_det_idx_tiny += len(current_detections_tiny)
            detections_tiny += current_detections_tiny
            ## ---
            mask = class_predictions[:, 0].gt(0.0)
            class_predictions = torch.masked_select(
                    class_predictions, mask.unsqueeze(-1)
                ).view(-1, 5).cpu().numpy()
            current_detections = _get_frame_detections(
                i, class_index, class_predictions, start_det_idx
            )

            start_det_idx += len(current_detections)
            detections += current_detections

     
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
       

    with MetricsDatabase.create(path=":memory:", mode="w") as db_out:
        db_out.export_images_iter(iterator=frames, ignore_duplicates=True)
        db_out.export_detections_iter(iterator=detections_huge, ignore_duplicates=True)
        db_out.export_groundtruths_iter(iterator=labels_huge, ignore_duplicates=True)

        metrics_wrapper = DetectionMetricsWrapper(
            database=db_out,
            # TODO(@williamz): allow this to be passed in.
            #configuration=evaluation_settings,
        )
        results_huge = metrics_wrapper.results()

        # TODO(@williamz): investigate delegating all printing to metrics code (with possible
        # changes upstream).
    
    with MetricsDatabase.create(path=":memory:", mode="w") as db_out:
        db_out.export_images_iter(iterator=frames, ignore_duplicates=True)
        db_out.export_detections_iter(iterator=detections_medium, ignore_duplicates=True)
        db_out.export_groundtruths_iter(iterator=labels_medium, ignore_duplicates=True)

        metrics_wrapper = DetectionMetricsWrapper(
            database=db_out,
            # TODO(@williamz): allow this to be passed in.
            #configuration=evaluation_settings,
        )
        results_medium = metrics_wrapper.results()

    with MetricsDatabase.create(path=":memory:", mode="w") as db_out:
        db_out.export_images_iter(iterator=frames, ignore_duplicates=True)
        db_out.export_detections_iter(iterator=detections_small, ignore_duplicates=True)
        db_out.export_groundtruths_iter(iterator=labels_small, ignore_duplicates=True)

        metrics_wrapper = DetectionMetricsWrapper(
            database=db_out,
            # TODO(@williamz): allow this to be passed in.
            #configuration=evaluation_settings,
        )
        results_small = metrics_wrapper.results()

    with MetricsDatabase.create(path=":memory:", mode="w") as db_out:
        db_out.export_images_iter(iterator=frames, ignore_duplicates=True)
        db_out.export_detections_iter(iterator=detections_small, ignore_duplicates=True)
        db_out.export_groundtruths_iter(iterator=labels_tiny, ignore_duplicates=True)

        metrics_wrapper = DetectionMetricsWrapper(
            database=db_out,
            # TODO(@williamz): allow this to be passed in.
            #configuration=evaluation_settings,
        )
        results_tiny = metrics_wrapper.results()

    
    print('---total')
    dm_output.print_average_precision(results)
    print('---huge')
    dm_output.print_average_precision(results_huge)
    print('---medium')
    dm_output.print_average_precision(results_medium)
    print('---small')
    dm_output.print_average_precision(results_small)
    print('---tiny')
    dm_output.print_average_precision(results_tiny)