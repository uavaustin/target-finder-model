xhost +"local:docker@"

sudo docker run --runtime=nvidia -ti --net=host --ipc=host -e DISPLAY=$DISPLAY \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-v $PWD:/host \
uavaustin/target-finder-model-env:xavier/bin/bash

xhost -