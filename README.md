# target-finder-model

> Contains files needed to create the model for
> [target-finder](https://github.com/uavaustin/target-finder)

[![CircleCI](https://circleci.com/gh/uavaustin/target-finder-model/tree/master.svg?style=svg)](https://circleci.com/gh/uavaustin/target-finder-model/tree/master)


## Usage

`TODO`

## Dev Instructions

1. Download `git clone https://github.com/uavaustin/target-finder-model`
2. Get Darknet
  * `cd target-finder-model && git clone https://github.com/AlexeyAB/darknet.git`
  * `cd darknet` and edit `MakeFile`
    * If CPU `AVX=1` `OPENMP=1` `LIBSO=1`
    * If GPU `CPU=1` `CUDNN=1` `LIBSO=1`
    * `make`
3. Download Assets `???`

## Testing

To run the tests, first install `tox`.

```sh
$ pip3 install tox
```

Now the tests can be run by simply calling:

```sh
$ tox
```

This takes care of installing the development dependencies in a Python virtual
environment, and then runs the build process but with only 100 of each shape
instead of the full batch. After, it runs quick unit tests to ensure the model
file is loading as expected.

To only build the shapes and run the training step, or run the unit tests, you
can run `tox -e model` and `tox -e unit`, respectively.

These tests are automatically run on CircleCI on each commit, and will attach
the shapes as a build artifact.

## Releases

Building the full model is managed with CircleCI (along with the testing
above). Note that the full-sized shape generation might take around 2 or more
hours, assuming a 4-core machine, and using 8 threads. This is parallelized in
the CI environment, giving faster builds.

Full builds are run on tags and the model is uploaded as a build artifact at
the end and pushed to GitHub Releases.
