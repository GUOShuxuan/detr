"""NVIDIA Autonomous Driving Dataset.

Written by Ismail Elezi
BoxParser written by William Zhang

Copyright (c) 2020 NVIDIA CORPORATION.  All rights reserved.
"""

from copy import deepcopy
import os
from pathlib import Path

from PIL import Image

# import sys
# sys.path.append("/home/shuxuang/debug/ai-infra/moduluspy")
# sys.path.append("/home/shuxuang/nvcode/ai-infra/")
from modulus.multi_task_loader.dataset import (
    LabelDataType,
    SqliteDataset,
)
from modulus.multi_task_loader.image_io import read_image
from modulus.multi_task_loader.task_parsers import apply_single_stm

import numpy as np
import torch
import torch.utils.data as data
import random
# import matplotlib.pyplot as plt ## add from Ismail
# import sys ## add from Ismail
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


import sandbox.williamz.detr.datasets.transforms as T
from sandbox.williamz.detr.datasets.nvidia_utils import ConvertCocoPolysToMask

import IPython

"""
NVIDIA_CLASSES = (
    "rider",
    "road_sign",
    "traffic_light",
    "automobile",
    "heavy_truck",
    "person",
    "hazard",
    "vehicle_group",
    "bicycle",
    "motorcycle",
    "stroller",
    "person_group",
    "misc_vehicle",
    "unclassifiable_vehicle",
    "headlight",
    "other_animal",
    "cycle_group",
    "taillight",
)"""

NVIDIA_CLASSES_MAPPED = {
    "automobile": "car",
    "bicycle": "bicycle",
    "heavy_truck": "car",
    "heavy truck": "car",
    "motorcycle": "bicycle",
    "person": "person",
    "person group": "person",
    "person_group": "person",
    "rider": "person",
    "road sign": "road_sign",
    "road_sign": "road_sign",
    "traffic light": "traffic_light",
    "traffic_light": "traffic_light",
    "unclassifiable vehicle": "car",
    "unclassifiable_vehicle": "car",
    "vehicle_group": "car",
    "stroller": "stroller",
    "hazard": "hazard",
    "headlight": "headlight",
    "misc_vehicle": "car",
    "other_animal": "other_animal",
    "cycle_group": "bicycle",
    "taillight": "taillight",
}

NVIDIA_CLASSES = (
    "car",
    "bicycle",
    "person",
    "road_sign",
    "traffic_light",
    "stroller",
    "hazard",
    "headlight",
    "other_animal",
    "taillight",
)

class_to_ind = dict(zip(NVIDIA_CLASSES, range(len(NVIDIA_CLASSES))))


class BoxParser:
    """Box parser."""

    def __init__(self):
        """Constructor."""
        # We will filter out rows that are not of this type, just to be sure,
        # Note that something similar could also be achieved by supplying
        # the appropriate SQL clause to `feature_conditions` when instantiating
        # the `SqliteDataset`.
        self._label_data_type = LabelDataType.from_string("SHAPE2D:BOX2D")

    def __call__(self, rows, frame, *args, **kwargs):
        """Call method.

        This is what `SqliteDataset.__getitem__` will be calling.

        Argumentss:
            rows (list): List of `Feature` tuples.
            frame (Frame): `Frame` tuple.

        Returns:
            image (np.array): CHW image.
            boxes (np.array): `np.float32` array of shape [num_objects * 4] containing
                the vertices of the bounding boxes. The order is (left, top, right,
                bottom).
            classes (list): List of class names (str).
        """
        # First, let's weed out potential uninteresting labels (as far as this parser
        # is concerned).
        rows = [
            row for row in rows if row.label_data_type == self._label_data_type
        ]

        boxes = np.array(
            [coord for feature in rows for coord in feature.data["vertices"]],
            dtype=np.float32,
        )
        class_names = [feature.label_name.strip().lower() for feature in rows]

        # This is because labels are in the original labeling space (typically 1920 x 1208),
        # but the export is typically a half res export (960 x 604) or a (960 x 604) center
        # crop.
        boxes = apply_single_stm(vertices=boxes, stm=frame.label_stm)

        image = read_image(
            image_path=frame.path, height=frame.original_height, width=frame.original_width
        )

        return image, boxes, class_names


class NVIDIADetection(data.Dataset):
    """NVIDIA Detection Dataset Object.

    Input is image, target is annotation.

    Arguments:
        supervised_indices (list): List of indices for which we use labels
        transform (Transform): The augmentation used
        pseudo_labels (list): List of indices for which we use pseudo-labels
    """

    def __init__(
        self,
        image_sets=None,
        transform=None,
        name="NVIDIA",
        image_sets2=None,
        mode='train'
    ):
        """Initialize the class."""
        parser = BoxParser()
        if mode == 'test':
            self.sqlite_dataset = SqliteDataset(
                filename=os.path.join(image_sets, "dataset.sqlite"),
                export_format_name="rgb_half-xavierisp",
                export_path=os.path.join(image_sets, "frames"),
                feature_parser=parser,
                exclude_frames="UNLABELED",
                feature_conditions=["features.label_data_type = 'BOX2D'"],                          
                # frame_conditions=["sequences.camera_location='forward center'"]
                # additional_conditions=["sequences.camera_location = 'forward_center'"]
            )
        else:
            self.sqlite_dataset = SqliteDataset(
                # filename=os.path.join(image_sets, "export.sqlite"),
                filename=os.path.join(image_sets, "dataset_300k.sqlite"),
                export_format_name="rgb_half-xavierisp",
                export_path=os.path.join(image_sets2, "frames"),
                feature_parser=parser,
                exclude_frames="UNLABELED",
                feature_conditions=["features.label_data_type = 'BOX2D'",
                                    # "sequences.camera_location = 'forward_center'"
                                    ], # "sequences.camera_location='%center%'"
                # frame_conditions=["sequences.camera_location='forward center'"]
                # additional_conditions=["sequences.camera_location = 'forward_center'"]
            )

        """Constructor."""
        self.transform = transform
        self.prepare = ConvertCocoPolysToMask()
        self.name = name
        self.class_to_ind = dict(zip(NVIDIA_CLASSES, range(len(NVIDIA_CLASSES))))

    def __getitem__(self, index):
        """Get an item in format image, label, semi (supervised, pseudo_labeled or unsupervised."""
        img, target = self.pull_item(index)
        return img, target

    def __len__(self):
        """Return the number of items in the dataset."""
        return len(self.sqlite_dataset)

    def pull_item(self, index):
        """Pull item."""
        img, ori_target = self.pull_image_and_anno(index)
#         print(img.shape) # (3, 604, 960)
#         print(target)
        # turn img from np.ndarray to PIL
        img = img.transpose(1,2,0)
        # to rgb
        img = img[:, :, (2, 1, 0)]
        img = Image.fromarray(np.uint8(img*255), mode='RGB')
        # img = Image.fromarray(np.uint8(img*255)).convert('RGB') # <PIL.Image.Image image mode=RGB size=960x604 at 0x7FB0479B4FD0>
#         print(img)
        anno = {'bbox': ori_target[:, :4], 
                'category_id': ori_target[:, 4]} # change the before 3 target to ori_target
        image_id = index
        target = {'image_id': image_id, 'annotations': anno}
#         print(target)
        img, target = self.prepare(img, target)
#         print(img)
#         print(target)
        # img = img.transpose(1, 2, 0)
        # height, width, _ = img.shape
        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target
        
    def pull_item_eval(self, index):
        """Pull item."""
        img, ori_target = self.pull_image_and_anno(index) # img: numpy.ndarray
#         print(img.shape) # (3, 604, 960)
#         print(target)
        # turn img from np.ndarray to PIL H*W*#
        img = img.transpose(1,2,0) # => (604, 960, 3) # img_orig
        # to rgb from bgr
        # IPython.embed()
        img = img[:, :, (2, 1, 0)]
        # ? how this influence the img quality? 
        
        # # try this?
        # # method2: img== img (before) True # have to rgb first
        # img = img_orig * 255
        # img = img.astype('uint8')
        # # to rgb
        # img[:, :, (2, 1, 0)]
        # img = Image.fromarray(img, mode='RGB') # does RGB mode turn the dimensions of ndarray?
        img = Image.fromarray(np.uint8(img*255), mode='RGB') #.convert('RGB') #PIL.Image.Image image mode=RGB size=960x604
        # before
        # img = Image.fromarray(np.uint8(img*255)).convert('RGB') # <PIL.Image.Image image mode=RGB size=960x604 at 0x7FB0479B4FD0>
#       img.save('/home/shuxuang/experiments/demos/img.jpg')
        anno = {'bbox': ori_target[:, :4], 
                'category_id': ori_target[:, 4]}
        image_id = index
        target = {'image_id': image_id, 'annotations': anno}
#         print(target)
        img, target = self.prepare(img, target)
#         print(img)
#         print(target)
        # img = img.transpose(1, 2, 0)
        # height, width, _ = img.shape
        if self.transform is not None:
            img, target = self.transform(img, target) # img_size: [3, 800, 1271]

        return img, ori_target, target

    def pull_item_vis(self, index):
        """Pull item."""
        img, ori_target = self.pull_image_and_anno(index) # img: numpy.ndarray
#         print(img.shape) # (3, 604, 960)
#         print(target)
        # turn img from np.ndarray to PIL H*W*#
        img = img.transpose(1,2,0) # => (604, 960, 3) # img_orig
        # orig_img_ = Image.fromarray(np.uint8(img*255), mode='RGB')
        # orig_img_.save('/home/shuxuang/experiments/demos/img_rgb.jpg')
        # plt.imsave('/home/shuxuang/experiments/demos/img_rgb-plt.jpg', np.uint8(img*255))
        # to rgb from bgr
        # IPython.embed()
        img = img[:, :, (2, 1, 0)]
        # ? how this influence the img quality? 
        
        # # try this?
        # # method2: img== img (before) True # have to rgb first
        # img = img_orig * 255
        # img = img.astype('uint8')
        # # to rgb
        # img = img[:, :, (2, 1, 0)]
        # img = Image.fromarray(img, mode='RGB') # does RGB mode turn the dimensions of ndarray?
        orig_img = Image.fromarray(np.uint8(img*255), mode='RGB') #.convert('RGB') #PIL.Image.Image image mode=RGB size=960x604
        # before
        # img = Image.fromarray(np.uint8(img*255)).convert('RGB') # <PIL.Image.Image image mode=RGB size=960x604 at 0x7FB0479B4FD0>
#       img.save('/home/shuxuang/experiments/demos/img.jpg')
        anno = {'bbox': ori_target[:, :4], 
                'category_id': ori_target[:, 4]}
        image_id = index
        target = {'image_id': image_id, 'annotations': anno}
#         print(target)
        img, target = self.prepare(orig_img, target)
#         print(img)
#         print(target)
        # img = img.transpose(1, 2, 0)
        # height, width, _ = img.shape

        if self.transform is not None:
            img, target = self.transform(img, target) # img_size: [3, 800, 1271]

        return orig_img_, img, ori_target, target

    
    def pull_image_and_anno(self, index):
        """Pull image and its annotation."""
        # Get an image, and its accompanying raster.
        image, boxes, class_names = self.sqlite_dataset[index]

        # for visualize and debug
        # image = image.transpose(1, 2, 0)
        # IPython.embed(header='pull image')
        # # print(image.shape)
        # print('bgr test')
        # img_bgr = np.uint8(image*255)
        # img_bgr = img_bgr[:,:, (2,1,0)]
        # print(
        #     'bgr finish'
        # )
        # plt.imsave('/home/shuxuang/experiments/demos/img_bgr-plt-inner.jpg', img_bgr)
        # sys.exit(0)
        assert boxes.size // 4 == len(class_names)

        # insert the boxes into an array of boxes with coordinates (x1, y1, x2, y2, class)
        len_boxes = len(boxes)
        new_boxes = np.zeros((len_boxes//2, 5))

        for i in range(0, len_boxes, 2):
            new_boxes[i//2, 0] = int(boxes[i, 0])
            new_boxes[i//2, 1] = int(boxes[i, 1])
            new_boxes[i//2, 2] = int(boxes[i+1, 0])
            new_boxes[i//2, 3] = int(boxes[i+1, 1])
        # adjusted_boxes = np.zeros_like(new_boxes)
        adjusted_boxes = deepcopy(new_boxes)
        adjusted_boxes[:, 0] = np.minimum(new_boxes[:, 0], new_boxes[:, 2])
        adjusted_boxes[:, 1] = np.minimum(new_boxes[:, 1], new_boxes[:, 3])
        adjusted_boxes[:, 2] = np.maximum(new_boxes[:, 0], new_boxes[:, 2])
        adjusted_boxes[:, 3] = np.maximum(new_boxes[:, 1], new_boxes[:, 3])

        for i, cl_name in enumerate(class_names):
            # adjusted_boxes[i, 4] = float(class_to_ind[cl_name])
            adjusted_boxes[i, 4] = float(class_to_ind[NVIDIA_CLASSES_MAPPED[cl_name]])
        return image, adjusted_boxes

    def target_transform(self, target, width, height):
        """Divide the coordinates by width and height."""
        target[:, 0] /= width
        target[:, 2] /= width
        target[:, 1] /= height
        target[:, 3] /= height
        return target


def make_coco_transforms(image_set):

# MEANS = (104, 117, 123) as bgr 
    normalize = T.Compose([
        T.ToTensor(),
        # T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # mean=(104, 117, 123) ==> (0.40784313725490196 0.4588235294117647 0.4823529411764706) transpose(2,1,0) = [0.482, 0.459, 0.408]
        # T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # coco mean and std
        T.Normalize([0.482, 0.459, 0.408], [1., 1., 1.]) # mean=(104, 117, 123) for not for rgb, bgr instead, after transpose [0.482, 0.459, 0.408]
    ])
    print('NVdata Norm: [0.482, 0.459, 0.408], [1., 1., 1.]')
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'test':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

def build_nvdataset(dataset_root, mode):
    # root0 = Path(dataset_root[0])
    # assert root0.exists(), f'provided NVData path {root0} does not exist'
    # root1 = Path(dataset_root[1])
    # assert root1.exists(), f'provided NVData path {root1} does not exist'
    dataset = NVIDIADetection(
            # supervised_indices=None,
            image_sets=dataset_root[0],
            # transform=SSDAugmentation(data_config["min_dim"], MEANS),
            transform=make_coco_transforms(mode),
            image_sets2=dataset_root[1],
            mode=mode
        )

    return dataset