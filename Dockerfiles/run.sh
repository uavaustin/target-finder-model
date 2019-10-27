xhost +"local:docker@"

if [ $# -eq 0 ] ; then
  sudo docker run --runtime=nvidia -ti \
    uavaustin/target-finder-model-env:latest /bin/bash
else 
  sudo docker run --runtime=nvidia -ti \
  -v $1:/host/mounted/ \
  uavaustin/target-finder-model-env:latest /bin/bash

fi
