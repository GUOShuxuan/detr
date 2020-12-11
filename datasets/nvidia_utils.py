"""Data module."""

import cv2
import numpy as np
import torch

import IPython

def detection_collate(batch):
    """Detection collate.

    Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    # changed when semi-supervised
    targets = []
    imgs = []
    semis = []
    sample = None
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
        if len(sample) == 3:
            semis.append(torch.FloatTensor(sample[2]))

    if sample is not None and len(sample) == 2:
        return torch.stack(imgs, 0), targets

    return torch.stack(imgs, 0), targets, semis


def base_transform(image, size, mean):
    """Base transform."""
    x = cv2.resize(image, (size, size)).astype(np.float32)
    x -= mean
    x = x.astype(np.float32)
    return x


class BaseTransform:
    """Base transform."""

    def __init__(self, size, mean):
        """Constructor."""
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        """Call."""
        return base_transform(image, self.size, self.mean), boxes, labels


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        # anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        # boxes = [obj["bbox"] for obj in anno]
        boxes = anno["bbox"]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        # boxes[:, 2:] += boxes[:, :2] #already been xyxy, for coco, top-left, w, h
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        # IPython.embed()
        # classes = [obj["category_id"] for obj in anno]
        classes = anno["category_id"]
        classes = torch.tensor(classes, dtype=torch.int64)


        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        target["iscrowd"] = torch.zeros_like(classes)

        return image, target