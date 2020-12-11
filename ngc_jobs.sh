ngc batch run --name "detr-base" --ace nv-us-west-2 --instance dgx1v.32g.4.norm --commandline "cd /workspace/code/detr ; sh envs.sh" --result result/ --image "nvidian/robotics/object-detection:mymmdetection-pytorch1.5" --org nvidian --team robotics --datasetid 67324:/dataset --workspace nvcode:/workspace/code:RW


ngc batch run --name "detr-base-4gpus" --ace nv-us-west-2 --instance dgx1v.32g.8.norm --commandline "cd /workspace/code/detr ; sh envs_4gpus.sh" --result result/ --image "nvidian/robotics/object-detection:mymmdetection-pytorch1.5" --org nvidian --team robotics --datasetid 67324:/dataset --workspace nvcode:/workspace/code:RW

ngc batch run --name "detr-base-8gpus" --ace nv-us-west-2 --instance dgx1v.32g.8.norm --commandline "cd /workspace/code/detr ; sh envs.sh" --result result/ --image "nvidian/robotics/object-detection:mymmdetection-pytorch1.5" --org nvidian --team robotics --datasetid 67324:/dataset --workspace nvcode:/workspace/code:RW


output_dir
remotePath": "/home/shuxuang/debug/detr/",