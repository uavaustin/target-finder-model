# Args for when running docker test container, meant to be overridden
DOCKERARGS :=

IMAGE_NAME := uavaustin/target-finder-model-env

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
	docker run --rm -v $(CUR_DIR)/../:/host $(IMAGE_NAME):test $(DOCKERARGS) 

.PHONY: clean
clean:
	docker rmi $(IMAGE_NAME):tf1
	docker rmi $(IMAGE_NAME):test
