#!/bin/bash -e

cd $(dirname "$0")

# Placeholder models for release generation. Mainly CI use. 
model_pkg="models-v1.tar.gz"
model_link="https://bintray.com/uavaustin/target-finder-assets/download_file?file_path=""$model_pkg"
od_model_dir="../target_finder_model/data/clf"
clf_model_dir="../target_finder_model/data/det"
config="../config.yaml"

# Check to see if model dirs exist.
if [ ! -d "$od_mode_dir" ] && [ ! -d "$clf_model_dir" ]; then
  echo "Downloading placeholder models."
  wget -q "$model_link" -O models-v1.tar.gz
  tar xzf models-v1.tar.gz
  mv models-v1/models/*.pb "../target_finder_model/data/"
  rm -rf models-v1
  rm "$model_pkg"
else 
  echo "Found existing models."
fi

# Find the version number to release.
version=$(grep -o -e "'.*'" "../target_finder_model/version.py" | tr -d "'")

echo "Detected version ""$version""."

pushd ..
tf_stage_dir="release/staging"
archive_name="target-finder-model-""$version"".tar.gz"

# Create the staging directory and the target-finder folder.
echo "Staging files."

# Copy over python files.
mkdir -p "$tf_stage_dir""/target_finder_model"
find "target_finder_model/" -name "*.py" -exec cp "{}" \
  "$tf_stage_dir/target_finder_model/" \;
# Copy model dirs
cp -r "target_finder_model/data" "$tf_stage_dir/target_finder_model/"
# Copy over configuration and informational files.
cp "README.md" "LICENSE" "setup.py" "$tf_stage_dir"

# Compress the directory.
echo "Creating archive."
pushd "release"

tar -C "staging" -czf "$archive_name" .

echo -e "\033[32mCreated target-finder-model release" \
  "(""$archive_name"")\033[0m"

# Remove the staging directory.
echo "Removing staging files"
rm -rf staging
popd
popd