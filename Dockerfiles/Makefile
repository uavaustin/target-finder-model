# Flags for docker when building images, meant to be overridden
DOCKERFLAGS :=

IMAGE_NAME := uavaustin/target-finder-model-env

IMAGE_TAG := latest

MOUNT_DIR := /

CUR_DIR := $(shell pwd)

.PHONY: all 
all: tf1 test

.PHONY: tf1
tf1:
	docker build -t $(IMAGE_NAME):tf1 -f Dockerfile.tf1 $(DOCKERFLAGS) . 
	
.PHONY: tf2
tf2:
	docker build -t $(IMAGE_NAME):tf2 -f Dockerfile.tf2 $(DOCKERFLAGS) .

.PHONY: test
test:
	docker build -t $(IMAGE_NAME):test -f Dockerfile.test $(DOCKERFLAGS) ..
	docker run -it --rm $(IMAGE_NAME):test

.PHONY: run
run:
	docker run --runtime=nvidia -ti -v $(MOUNT_DIR):/host/mounted/ $(IMAGE_NAME):$(IMAGE_TAG) /bin/bash

.PHONY: test-run
test-run:
	docker run -ti $(IMAGE_NAME):test /bin/bash
