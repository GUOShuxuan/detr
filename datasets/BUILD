# Copyright (c) 2020 NVIDIA CORPORATION.  All rights reserved.

load("@pip_deps//:requirements.bzl", "requirement")
load("//ci/runtime_resources:extension.bzl", "py_static_analysis")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "models",
    srcs = ["__init__.py"],
    deps = [
        requirement("torch"),
        requirement("torchvision"),
    ],
)

py_library(
    name = "nvidia",
    srcs = ["nvidia.py"],
    deps = [
        "//moduluspy/modulus/multi_task_loader",
        "//sandbox/williamz/detr/datasets:transforms",
        "//sandbox/williamz/detr/datasets:nvidia_utils",
        requirement("matplotlib"),
        requirement("numpy"),
        requirement("opencv-python"),
        requirement("torch"),
        requirement("ipython"),
    ],
)

py_library(
    name = "subset_sequential_sampler",
    srcs = ["subset_sequential_sampler.py"],
    deps = [
        requirement("torch"),
    ],
)

py_library(
    name = "nvidia_utils",
    srcs = ["nvidia_utils.py"],
    deps = [
        requirement("numpy"),
        requirement("opencv-python"),
        requirement("torch"),
        requirement("ipython"),
    ],
)

py_library(
    name = "transforms",
    srcs = ["transforms.py"],
    deps = [
        "//sandbox/williamz/detr/util",
        "//sandbox/williamz/detr/util:box_ops",
        "//sandbox/williamz/detr/util:misc",
        requirement("torch"),
        requirement("torchvision"),
        requirement("pillow"),
    ],
)

