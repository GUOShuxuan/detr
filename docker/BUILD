# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
load("//ci/runtime_resources:extension.bzl", "buildifier_test")

package(default_visibility = ["//sandbox/williamz:__subpackages__"])

# Docker / launcher related stuff.
load("//maglev_sdk/docker_container:rules.bzl", "container_image_py")
load("//maglev_sdk/launcher:launcher_rules.bzl", "launcher_run")
load("@io_bazel_rules_docker//python:image.bzl", "py_layer")

APPS = [
    "//sandbox/williamz/detr:main",
    "//sandbox/williamz/detr:train_with_eval",
    "//sandbox/williamz/detr:train_with_eval_init",
    "//sandbox/williamz/detr:eval_vis",
]

# Docker / launcher related stuff.
# Select all modulus dependencies of apps.
py_layer(
    name = "detr_layer",
    filter = "//sandbox/williamz/detr/",
    deps = APPS,
)

# Select all external dependencies (pip packages) of apps.
py_layer(
    name = "external_layer",
    filter = "@",
    deps = APPS,
)

container_image_py(
    name = "image",
    add_to_path = [
        "sandbox/williamz/detr",
    ],
    layers = [
        ":external_layer",
        ":detr_layer",
    ],
    deps = APPS,
)

launcher_run(
    name = "run",
    image = ":image",
)

buildifier_test()
