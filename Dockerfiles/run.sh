xhost +"local:docker@"

if [ $# -eq 0 ] ; then
  sudo docker run --runtime=nvidia -ti --net=host --ipc=host -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    uavaustin/target-finder-model-env:latest /bin/bash
else 
  sudo docker run --runtime=nvidia -ti --net=host --ipc=host -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $1:/host/mounted/ \
  uavaustin/target-finder-model-env:latest /bin/bash

fi
