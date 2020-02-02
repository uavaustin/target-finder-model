#!/bin/bash
#
# Reference: https://docs.bazel.build/versions/master/install-ubuntu.html#install-with-installer-ubuntu

set -e

folder=${HOME}/src
mkdir -p $folder

echo "** Download bazel-0.26.1 sources"
cd $folder
if [ ! -f bazel-0.26.1-dist.zip ]; then
  wget https://github.com/bazelbuild/bazel/releases/download/0.26.1/bazel-0.26.1-dist.zip
fi

echo "** Build and install bazel-0.26.1"
unzip bazel-0.26.1-dist.zip -d bazel-0.26.1-dist
cd bazel-0.26.1-dist

./compile.sh
sudo cp output/bazel /usr/local/bin
echo "** Build bazel-0.26.1 successfully"
