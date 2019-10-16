# Flags for docker when building images, meant to be overridden
DOCKERFLAGS :=

IMAGE_NAME := uavaustin/target-finder-model-env

IMAGE_TAG := latest

MOUNT_DIR := /

.PHONY: all
all: image

.PHONY: image
image:
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) -f Dockerfile $(DOCKERFLAGS) .

.PHONY: run
run:
	docker run --runtime=nvidia -ti -v $(MOUNT_DIR):/host/mounted/ $(IMAGE_NAME):$(IMAGE_TAG) /bin/bash

.PHONY: clean
clean:
	docker rmi --force $(IMAGE_NAME):$(IMAGE_TAG)
