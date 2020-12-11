# Copyright (c) 2020 NVIDIA CORPORATION.  All rights reserved.

load("@pip_deps//:requirements.bzl", "requirement")
load("//ci/runtime_resources:extension.bzl", "py_static_analysis")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "util",
    srcs = ["__init__.py"],
    deps = [
        ":box_ops",
        ":misc",
        ":plot_utils",
    ],
)

py_library(
    name = "box_ops",
    srcs = ["box_ops.py"],
    deps = [
         requirement("torch"),
         requirement("torchvision"),
    ],
)

py_library(
    name = "misc",
    srcs = ["misc.py"],
    deps = [
         requirement("torch"),
         requirement("torchvision"),
    ],
)

py_library(
    name = "plot_utils",
    srcs = ["plot_utils.py"],
    deps = [
         requirement("torch"),
         requirement("pandas"),
         requirement("numpy"),
         requirement("matplotlib"),
    ],
)