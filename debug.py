# from detr.subset_seque import SubsetSequentialSampler
# from .nvidia_utils import detection_collate
# from datasets import nvidia_utils
import argparse
from copy import deepcopy ## add from Ismail
import os
import random
from statistics import mean, pstdev
import time
import numpy as np
import torch.utils.data as data
import IPython

from detr.datasets.nvidia_utils import BaseTransform, detection_collate
from torch.utils.data.sampler import SubsetRandomSampler
from detr.datasets.nvidia import (
    NVIDIADetection,
)
# import pycocotools
# import submitit
from detr.datasets.nvidia import build_nvdataset
 
if __name__ == '__main__':
    # os.fil
    # train_dataset= build_nvdataset(
    #                     # indices=np.load('/home/shuxuang/datasets/id_1_criterion_Max_SSD_num_labels_50000.npy'),
    #                     dataset_root=['/home/shuxuang/datasets/annotation_sql_nvidia', '/home/shuxuang/datasets/frames_nvidia'],
    #                     # batch_size=2,
    #                     # num_workers=2,
    #                     data_config=None,  # <-- this really should be refactored.
    # )
    train_dataset=build_nvdataset(dataset_root=['~/datasets/annotation_sql_nvidia', '~/datasets/frames_nvidia'], 
                                mode='train')
    # import ipdb; ipdb.set_trace()
    # IPython.embed()
    print(len(train_dataset))
    print(train_dataset[0])

# dazel run //detr/datasets:nvidia  
# dazel run //detr:debug