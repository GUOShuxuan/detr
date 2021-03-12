# Copyright (c) 2020 NVIDIA CORPORATION.  All rights reserved.

load("@pip_deps//:requirements.bzl", "requirement")
load("//ci/runtime_resources:extension.bzl", "py_static_analysis")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "models",
    srcs = ["__init__.py"],
    deps = [
        ":detr"
    ],
)

py_library(
    name = "backbone",
    srcs = ["backbone.py"],
    deps = [
        ":position_encoding",
        "//sandbox/williamz/detr/util:misc",
       # from sandbox.ptpoc.utils import deserialize_object, load_spec
        "//sandbox/ptpoc/utils",
        "//sandbox/ptpoc",
        requirement("torch"),
        requirement("torchvision"),
        requirement('ipython'),
    ],
)

py_library(
    name = "detr",
    srcs = ["detr.py"],
    deps = [
        ":backbone",
        ":matcher",
        ":segmentation",
        ":transformer",
        "//sandbox/williamz/detr/util",
        "//sandbox/williamz/detr/util:misc",
        requirement("torch"),
    ],
)

py_library(
    name = "matcher",
    srcs = ["matcher.py"],
    deps = [
        "//sandbox/williamz/detr/util",
        "//sandbox/williamz/detr/util:box_ops",
        requirement("torch"),
        requirement("scipy"),
    ],
)

py_library(
    name = "position_encoding",
    srcs = ["position_encoding.py"],
    deps = [
        "//sandbox/williamz/detr/util",
        "//sandbox/williamz/detr/util:misc",
        requirement("torch"),
    ],
)

py_library(
    name = "segmentation",
    srcs = ["segmentation.py"],
    deps = [
        "//sandbox/williamz/detr/util",
        "//sandbox/williamz/detr/util:misc",
        "//sandbox/williamz/detr/util:box_ops",
        requirement("torch"),
        requirement("pillow"),
    ],
)

py_library(
    name = "transformer",
    srcs = ["transformer.py"],
    deps = [
        requirement("torch"),
    ],
)