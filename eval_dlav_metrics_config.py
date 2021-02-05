# Copyright (c) 2020 NVIDIA CORPORATION.  All rights reserved.

"""Evaluation script that uses dlav/metrics/detection implementation of mAP."""

import argparse
import os
import time

from dlav.metrics.detection.data.metrics_database import MetricsDatabase
# from dlav.metrics.detection.process.detection_metrics_wrapper import DetectionMetricsWrapper
from dlav.metrics.detection.report import output as dm_output
import torch
# import matplotlib
# from matplotlib import pyplot as plt

from sandbox.williamz.detr.datasets.nvidia import (
    NVIDIA_CLASSES,
    NVIDIADetection,
)

from sandbox.williamz.secret_project.types import Detection, Label
from sandbox.williamz.detr.detection_metrics_wrapper import DetectionMetricsWrapper

from dlav.drivenet.evaluation.evaluation_config import build_evaluation_config
from dlav.drivenet.spec_handling.spec_loader import load_experiment_spec

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


  


def evaluate(model, dataset, postprocessors, device, out_dir=None, ):
    """Main function."""
    # The +1 for num_classes is for the background class.
    # This line is needed because the model is typically trained with
    # torch.nn.DataParallel, which prefixes state keys with "module".
    # TODO(@williamz): Write a util that allows loading the state dict
    # regardless of whether the model has been wrapped by DataParallel. 
    experiment_spec_file = '/home/shuxuang/experiments/drivenet_evaluate/evaluate_active_learning.txt'
    experiment_spec = load_experiment_spec(experiment_spec_file)

    evaluation_config = build_evaluation_config(experiment_spec.evaluation_config)
    
    model.eval()
  
    # These will be passed to dlav/metrics/detection, and will accumulate the data in
    # a format compatible with that code.
    frames, labels, detections = [], [], []
    # TODO(@williamz): Switch to batched mode once the `pull_item` has been put
    # into `__getitem__` directly.
    for i in range(10):  # need to implement returning the length of the dataset #len(dataset)
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
            # configuration=evaluation_config,
            # config=evaluation_config.evaluation_bucket_config
            evaluation_config=evaluation_config
        )
        IPython.embed()
        results = metrics_wrapper.results()

        # IPython.embed()
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



