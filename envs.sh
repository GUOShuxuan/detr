tar -xvf /dataset/coco2017zip.tar.gz -C /raid/
unzip /raid/coco2017zip/annotations_trainval2017.zip -d /raid/coco2017
unzip -q /raid/coco2017zip/train2017.zip -d /raid/coco2017
unzip -q /raid/coco2017zip/test2017.zip -d /raid/coco2017
unzip /raid/coco2017zip/val2017.zip -d /raid/coco2017

conda install cython scipy
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

apt-get update -qq && \
    apt-get install -y git vim libgtk2.0-dev && \
    rm -rf /var/cache/apk/*

pip --no-cache-dir install Cython

pip --no-cache-dir install -r /workspace/requirements.txt


# python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --coco_path /raid/coco2017
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --lr_drop 100 --epochs 150 --coco_path /raid/coco2017 --output_dir ./results/8gpus/