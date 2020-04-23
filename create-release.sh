#!/bin/sh -e

cd $(dirname "$0")
pwd
config="config.yaml"

# Find the version number to release.
version=$(grep -o -e "'.*'" "target_finder_model/version.py" | tr -d "'")

echo "Detected version ""$version"

tf_stage_dir="release/staging/target-finder-model"
archive_name="target-finder-model-""$version"".tar.gz"

# Create the staging directory and the target-finder folder.
echo "Staging files"
mkdir -p "$tf_stage_dir""/target_finder_model/data/"
cp -r "target_finder_model/data/" "$tf_stage_dir""/target_finder_model/"

# Copy over python files.
mkdir -p "$tf_stage_dir""/target_finder_model"
find "target_finder_model/" -name "*.py" -exec cp "{}" \
  "$tf_stage_dir/target_finder_model/" \;

# Copy over configuration and informational files.
cp README.md LICENSE \
  setup.py "$tf_stage_dir"

# Compress the directory.
echo "Creating archive"

pushd "release/staging"
tar -czvf "$archive_name" "target-finder-model"
mv "$archive_name" ..

# Remove the staging directory.
echo "Removing staging files"
popd
rm -rf "release/staging"

echo -e "\033[32mCreated target-finder-model release" \
  "(""$archive_name"")\033[0m"
