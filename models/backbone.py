# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict
import os

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding

## adding for autonet
from sandbox.ptpoc.utils import deserialize_object, load_spec
import IPython

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name: # train_backbone is True, layer1 not train, while train others
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        # IPython.embed()
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)  
        # input:  torch.Size([2, 3, 604, 960])
        #xs['0'].size(): torch.Size([2, 2048, 19, 30]) 'orderdict'
        # IPython.embed() 
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask) #x.size():torch.Size([2, 2048, 19, 30]) mask.size():[2, 19, 30])
        # IPython.embed() 
        return out

class BackboneAutoNetBase(BackboneBase):
    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super(BackboneBase, self).__init__()
        for name, parameter in backbone.named_parameters():
            # print(name)
            if not train_backbone: #or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        # IPython.embed()
        # if return_interm_layers:
        #     return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        # else:
        #     return_layers = {'layer4': "0"}
        self.body = backbone
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)  #torch.Size([2, 256, 38, 60])
        # IPython.embed() 
        xs = {'0': xs}
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask) # torch.Size([2, 256, 38, 60])
        # IPython.embed() 
        return out

class BackboneAutoNet(BackboneAutoNetBase):
     # add autonet backbone
     def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool,
                 training_spec: str,
                 auto_checkpoint: str
                 ):
        if name.startswith('autonet'):     
            # training_spec = 'sandbox/williamz/detr/res_autonet/autonet_training_spec.yaml'
            # training_spec = os.path.join(os.environ["HOME"],'datasets/specs/autonet_training_spec.yaml')
            training_spec = load_spec(training_spec)
            model = deserialize_object(training_spec["model"])
            # autonet checkpoint 
            # checkpoint = 'sandbox/williamz/detr/res_autonet/final_epoch.checkpoint'
            checkpoint = auto_checkpoint
            # checkpoint = os.path.join(os.environ["HOME"],'datasets/autonet/final_epoch.checkpoint')
            if checkpoint is not None and os.path.isfile(checkpoint) and is_main_process():
                print(f'---------- Loading checkpoint for AutoNet -----')
                loaded_states = torch.load(checkpoint)
                model_state = loaded_states["model_state"]
                model.load_state_dict(model_state, strict=False)
                # backbone = model
            else:
                print(f'---------- No checkpoint for AutoNet -----')
            
            # get drivenet
            # IPython.embed()
            modules = []
            for block in model._blocks:
                if 'drive2d' in block["task_name"]:
                    modules.append(getattr(model, block['name']))
            backbone = nn.Sequential(*modules[:-1])
            num_channels = 256
            super().__init__(backbone, train_backbone, num_channels, return_interm_layers)

class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        if name.startswith('resnet'):
            backbone = getattr(torchvision.models, name)(
                replace_stride_with_dilation=[False, False, dilation],
                # pretrained=False, norm_layer=FrozenBatchNorm2d)
                pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
            num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
            super().__init__(backbone, train_backbone, num_channels, return_interm_layers)

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    if args.backbone.startswith('resnet'):
        backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    elif args.backbone.startswith('autonet'):
         backbone = BackboneAutoNet(args.backbone, train_backbone, return_interm_layers, args.dilation, args.training_spec, args.auto_checkpoint)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model


# debug test for drivenet part 
# inputs = torch.rand((4, 3, 544, 960))
# out = model(inputs)
# out.keys(): dict_keys(['drive2d', 'openroad', 'map', 'wait_sky'])
# out['drive2d'].keys(): ['1_cycle', '1_person', '1_vehicle']
# out['drive2d']['1_cycle'].keys(): dict_keys(['cov', 'bbox']) # 1, 4
# bbox = out['drive2d']['1_cycle']['bbox']
# cov = out['drive2d']['1_cycle']['cov']
# # module = getattr(model, 'drive2d')
# # drive2d = getattr(model, 'drive2d')
# # rebuild drive2d
# modules = []
# for block in model._blocks:
#     if 'drive2d' in block["task_name"]:
#         modules.append(getattr(model, block['name']))

# drivenet = nn.ModuleList(modules)
# f = open('/home/shuxuang/experiments/demos/detection-f/drive2d.txt', 'w')
# f.write(str(drivenet))
# # drivenet[0](inputs).size(): [4, 64, 136, 240]
# #  drivenet[1](drivenet[0](inputs)).size(): ([4, 256, 34, 60])
# # dout = drivenet[2](drivenet[1](drivenet[0](inputs)))
# #  d_bbox = dout['1_cycle']['bbox']
# d_cov = dout['1_cycle']['cov']
# torch.all(torch.eq(bbox, d_bbox)) #True
# torch.all(torch.eq(cov, d_cov)) # True

# drivnet = nn.Sequential(*modules)
# ddout = drivnet(inputs)
# dd_bbox = ddout['1_cycle']['bbox']
# dd_cov = ddout['1_cycle']['cov']
