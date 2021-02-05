# Copyright (c) 2020 NVIDIA CORPORATION.  All rights reserved.

load("@pip_deps//:requirements.bzl", "requirement")
load(
    "//ci/runtime_resources:extension.bzl",
    "buildifier_test",
    "py_import_test",
    "py_static_analysis",
    "pytest_test",
    "py_notebook",
)

package(default_visibility = ["//visibility:public"])

py_binary(
    name = "engine",
    srcs = ["engine.py"],
    deps = [
        "//sandbox/williamz/detr/util:misc",
        requirement("torch"),
    ],
)

py_binary(
    name = "engine_amp",
    srcs = ["engine_amp.py"],
    deps = [
        "//sandbox/williamz/detr/util:misc",
        requirement("torch"),
    ],
)

py_binary(
    name = "hubconf",
    srcs = ["hubconf.py"],
    deps = [
        "//sandbox/williamz/detr/models",
        "//sandbox/williamz/detr/models:backbone",
        "//sandbox/williamz/detr/models:detr",
        "//sandbox/williamz/detr/models:position_encoding",
        "//sandbox/williamz/detr/models:segmentation",
        "//sandbox/williamz/detr/models:transformer",    
        requirement("torch"),
    ],
)

py_binary(
    name = "main",
    srcs = ["main.py"],
    deps = [
        ":engine", 
        ":eval_dlav_metrics", 
        # "//moduluspy/modulus/multi_task_loader",
        "//sandbox/williamz/detr/util:misc",
        "//sandbox/williamz/detr/models",
        "//sandbox/williamz/detr/datasets:nvidia",
        requirement("torch"),
        requirement("numpy"),
        requirement("ipython"),
    ],
)

py_binary(
    name = "main_5classes",
    srcs = ["main_5classes.py"],
    deps = [
        ":engine", 
        ":eval_dlav_metrics", 
        # "//moduluspy/modulus/multi_task_loader",
        "//sandbox/williamz/detr/util:misc",
        "//sandbox/williamz/detr/models",
        "//sandbox/williamz/detr/datasets:nvidia_5classes",
        requirement("torch"),
        requirement("numpy"),
        requirement("ipython"),
    ],
)

py_binary(
    name = "main_amp",
    srcs = ["main_amp.py"],
    deps = [
        ":engine_amp", 
        ":eval_dlav_metrics", 
        # "//moduluspy/modulus/multi_task_loader",
        "//sandbox/williamz/detr/util:misc",
        "//sandbox/williamz/detr/models",
        "//sandbox/williamz/detr/datasets:nvidia",
        requirement("torch"),
        requirement("numpy"),
        requirement("ipython"),
    ],
)

py_binary(
    name = "test_all",
    srcs = ["test_all.py"],
    deps = [
        ":hubconf",
        "//sandbox/williamz/detr/models",
        "//sandbox/williamz/detr/models:matcher",
        "//sandbox/williamz/detr/models:position_encoding",
        "//sandbox/williamz/detr/models:backbone",
        "//sandbox/williamz/detr/util",
        "//sandbox/williamz/detr/util:misc",  
        requirement("torch"),
        requirement("numpy"),
    ],
)

py_binary(
    name = "debug",
    srcs = ["debug.py"],
    deps = [
       # "//detr/datasets",
        "//sandbox/williamz/detr/datasets:nvidia",
        "//sandbox/williamz/detr/datasets:nvidia_utils",
        requirement("torch"),
        requirement("numpy"),
        requirement("ipython"),
        requirement("torchvision"),
    ],
)

py_library(
    name = "detection_metrics_wrapper",
    srcs = ["detection_metrics_wrapper.py"],
    deps = [
        "//dlav/metrics/detection/config:constants",
        "//dlav/metrics/detection/data:metrics_database",
        "//dlav/metrics/detection/data:types",
        "//dlav/metrics/detection/data:drivenet_metrics_database",
       "//dlav/metrics/detection/data:yaml_types",
        "//dlav/metrics/detection/devkit:detection_metrics",
        "//dlav/metrics/detection/process:algorithms",
        "//dlav/metrics/detection/process:database_marshaller",
        "//dlav/metrics/detection/report:detection_results",
        "//dlav/metrics/detection/utilities:utils",
        requirement("numpy"),
        requirement("six"),
    ],
)

py_binary(
    name = "eval_dlav_metrics",
    srcs = ["eval_dlav_metrics.py"],
    deps = [
        "//sandbox/williamz/secret_project:types",
        "//sandbox/williamz/detr:detection_metrics_wrapper",
        "//dlav/metrics/detection/data:metrics_database",
        "//dlav/metrics/detection/report:output",
	    "//sandbox/williamz/detr/datasets:nvidia",
        requirement("torch"),
        requirement("ipython"),
        requirement("matplotlib"),
    ],
)

py_binary(
    name = "eval_dlav_metrics_config",
    srcs = ["eval_dlav_metrics_config.py"],
    deps = [
        "//sandbox/williamz/secret_project:types",
        "//sandbox/williamz/detr:detection_metrics_wrapper",
        "//dlav/metrics/detection/data:metrics_database",
        "//dlav/metrics/detection/report:output",
        "//dlav/drivenet/evaluation:evaluation_config",
        "//dlav/drivenet/spec_handling:spec_loader",
	    "//sandbox/williamz/detr/datasets:nvidia",
        requirement("torch"),
        requirement("ipython"),
        requirement("matplotlib"),
    ],
)

py_binary(
    name = "eval_dlav_vis",
    srcs = ["eval_dlav_vis.py"],
    deps = [
        "//sandbox/williamz/secret_project:types",
        "//sandbox/williamz/detr:detection_metrics_wrapper",
        "//dlav/metrics/detection/data:metrics_database",
        "//dlav/metrics/detection/report:output",
	    "//sandbox/williamz/detr/datasets:nvidia",
        requirement("torch"),
        requirement("ipython"),
        requirement("matplotlib"),
    ],
)

py_binary(
    name = "train_with_eval",
    srcs = ["train_with_eval.py"],
    deps = [
        ":engine", 
        ":eval_dlav_metrics", 
        # "//moduluspy/modulus/multi_task_loader",
        "//sandbox/williamz/detr/util:misc",
        "//sandbox/williamz/detr/models",
        "//sandbox/williamz/detr/datasets:nvidia",
        requirement("torch"),
        requirement("numpy"),
        requirement("ipython"),
    ],
)

py_binary(
    name = "eval",
    srcs = ["eval.py"],
    deps = [
        ":engine", 
        ":eval_dlav_metrics", 
        ":eval_dlav_metrics_config", 
        # "//moduluspy/modulus/multi_task_loader",
        "//sandbox/williamz/detr/util:misc",
        "//sandbox/williamz/detr/models",
        "//sandbox/williamz/detr/datasets:nvidia",
        requirement("torch"),
        requirement("numpy"),
        requirement("ipython"),
    ],
)

py_binary(
    name = "eval_5classes",
    srcs = ["eval_5classes.py"],
    deps = [
        ":engine", 
        ":eval_dlav_metrics", 
        ":eval_dlav_metrics_config", 
        # "//moduluspy/modulus/multi_task_loader",
        "//sandbox/williamz/detr/util:misc",
        "//sandbox/williamz/detr/models",
        "//sandbox/williamz/detr/datasets:nvidia_5classes",
        requirement("torch"),
        requirement("numpy"),
        requirement("ipython"),
    ],
)

py_binary(
    name = "train_with_eval_init",
    srcs = ["train_with_eval_init.py"],
    deps = [
        ":engine", 
        ":eval_dlav_metrics", 
        # "//moduluspy/modulus/multi_task_loader",
        "//sandbox/williamz/detr/util:misc",
        "//sandbox/williamz/detr/models",
        "//sandbox/williamz/detr/datasets:nvidia",
        requirement("torch"),
        requirement("numpy"),
        requirement("ipython"),
    ],
)

py_binary(
    name = "eval_vis",
    srcs = ["eval_vis.py"],
    deps = [
        ":engine", 
        ":eval_dlav_vis", 
        # "//moduluspy/modulus/multi_task_loader",
        "//sandbox/williamz/detr/util:misc",
        "//sandbox/williamz/detr/models",
        "//sandbox/williamz/detr/datasets:nvidia",
        requirement("torch"),
        requirement("numpy"),
        requirement("ipython"),
        requirement("matplotlib"),
    ],
)


py_import_test(
    "main",
    dotted_name = "sandbox.williamz.detr.main",
)

py_import_test(
    "main_amp",
    dotted_name = "sandbox.williamz.detr.main_amp",
)

py_import_test(
    "main_5classes",
    dotted_name = "sandbox.williamz.detr.main_5classes",
)

py_import_test(
    "eval",
    dotted_name = "sandbox.williamz.detr.eval",
)

py_import_test(
    "eval_vis",
    dotted_name = "sandbox.williamz.detr.eval_vis",
)

py_import_test(
    "train_with_eval",
    dotted_name = "sandbox.williamz.detr.train_with_eval",
)

py_import_test(
    "train_with_eval_init",
    dotted_name = "sandbox.williamz.detr.train_with_eval_init",
)

py_static_analysis(
    name = "static_test",
    python_module = "sandbox",
)
