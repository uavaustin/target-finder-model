FROM tensorflow/tensorflow:1.13.2-gpu-py3

WORKDIR /sources

RUN apt-get update \
    && apt-get install -y \
	git \
        protobuf-compiler \ 
    && rm -rf /var/lib/apt/lists/*

COPY Dockerfiles/requirements-docker.txt .

COPY Dockerfiles/entrypoint.sh /entrypoint.sh

RUN apt install python3-pip \
  && pip3 install --upgrade pip \
  && pip3 install tox

RUN pip3 install -r requirements-docker.txt

RUN git clone https://github.com/tensorflow/models.git

RUN git clone https://github.com/cocodataset/cocoapi.git \
    && cd cocoapi/PythonAPI \
    && make \
    && cp -r pycocotools /sources/models/research

WORKDIR /sources/models/research 

RUN protoc object_detection/protos/*.proto --python_out=.

ENV PYTHONPATH=$PYTHONPATH:/sources/models/research:/sources/models/research/slim

RUN python setup.py install --user

WORKDIR /host/mounted

ENTRYPOINT ["/entrypoint.sh"]
