# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
load("//maglev_sdk/workflow:workflow_rules.bzl", "maglev_workflow")

[maglev_workflow(
    images = ["//sandbox/williamz/detr/docker:image"],
    spec = spec,
) for spec in glob(["*.yaml", "*.yml"])]

