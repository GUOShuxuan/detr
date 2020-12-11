mkdir /workspace/dataset/
tar -xvf /dataset/coco2017zip.tar.gz -C /workspace/dataset/
unzip /workspace/dataset/coco2017zip/annotations_trainval2017.zip -d /workspace/dataset/coco2017
unzip -q /workspace/dataset/coco2017zip/train2017.zip -d /workspace/dataset/coco2017
unzip -q /workspace/dataset/coco2017zip/test2017.zip -d /workspace/dataset/coco2017
unzip /workspace/dataset/coco2017zip/val2017.zip -d /workspace/dataset/coco2017

conda install cython scipy
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

apt-get update -qq && \
    apt-get install -y git vim libgtk2.0-dev && \
    rm -rf /var/cache/apk/*

pip --no-cache-dir install Cython

pip --no-cache-dir install -r /workspace/requirements.txt


python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --coco_path /workspace/dataset/coco2017/ --output_dir ./results/4gpus/