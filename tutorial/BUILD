# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
load("//ci/runtime_resources:extension.bzl", "py_notebook")
load("@pip_deps//:requirements.bzl", "requirement")

py_notebook(
    name = "demo",
    srcs = ["demo.ipynb"],
    deps = [
        "//sandbox/williamz/detr:eval",
        "//moduluspy/modulus/multi_task_loader",
        "//sandbox/williamz/detr/models",
        "//sandbox/williamz/detr/datasets:nvidia",
        requirement("matplotlib"),
        requirement("numpy"),
        requirement("opencv-python"),
        requirement("torch"),
        requirement("torchvision"),
    ],
)

py_notebook(
    name = "drivenet_inference",
    srcs = ["drivenet_inference.ipynb"],
    deps = [
        "//sandbox/williamz/detr:eval",
        "//moduluspy/modulus/multi_task_loader",
        "//dlav/drivenet/model:simple_inferencer", 
        "//sandbox/williamz/detr/models",
        "//sandbox/williamz/detr/datasets:nvidia",
        requirement("matplotlib"),
        requirement("numpy"),
        # requirement("pickle"), 
        requirement("opencv-python"),
        requirement("torch"),
        requirement("torchvision"),
    ],
)