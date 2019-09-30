FROM tensorflow/tensorflow:latest-gpu-py3

WORKDIR /sources

RUN apt-get update \
    && apt-get install -y \
	git \
        protobuf-compiler \ 
    && rm -rf /var/lib/apt/lists/*

COPY Dockerfiles/requirements.txt .


RUN apt install python3-pip 
RUN pip3 install -r requirements.txt 

RUN git clone https://github.com/tensorflow/models.git

RUN git clone https://github.com/cocodataset/cocoapi.git \
    && cd cocoapi/PythonAPI \
    && make \
    && cp -r pycocotools /sources/models/research

WORKDIR /sources/models/research 

RUN protoc object_detection/protos/*.proto --python_out=.

ENV PYTHONPATH=$PYTHONPATH:/sources/models/research:/sources/models/research/slim
