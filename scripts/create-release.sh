#!/bin/sh -e

cd $(dirname "$0")

# Make sure data folder exits 
mkdir -p "../target_finder_model/data/"

# Download the placeholder frozen models
model_pkg="models-v1.tar.gz"
model_link="https://bintray.com/uavaustin/target-finder-assets/download_file?file_path=""$model_pkg"
od_model="../target_finder_model/data/det.pb"
clf_model="../target_finder_model/data/clf.pb"
config="../config.yaml"

if [ ! -f "$od_model" ] && [ ! -f "$cld_model" ]; then
  echo "Downloading placeholder models."
  wget "$model_link" -O models-v1.tar.gz
  tar xzf models-v1.tar.gz
  ls
  mv models-v1/models/*.pb "../target_finder_model/data/"
  rm -rf models-v1
  rm "$model_pkg"
fi

# Check that the model files exist.
[ -f "$od_model" ] || (>&2 echo "Missing Detection Model" && exit 1)
[ -f "$clf_model" ] || (>&2 echo "Missing Classification Model" && exit 1)

# Find the version number to release.
version=$(grep -o -e "'.*'" "../target_finder_model/version.py" | tr -d "'")

echo "Detected version ""$version"

tf_stage_dir="../release/staging/target-finder-model"
archive_name="target-finder-model-""$version"".tar.gz"

# Create the staging directory and the target-finder folder.
echo "Staging files"
mkdir -p "$tf_stage_dir""/target_finder_model/data/"
cp -r "../target_finder_model/data/" "$tf_stage_dir""/target_finder_model/"
cp "$od_model" "$tf_stage_dir""/target_finder_model/data/"
cp "$clf_model" "$tf_stage_dir""/target_finder_model/data/"
cp "$config" "$tf_stage_dir""/target_finder_model/data/"



# Copy over python files.
mkdir -p "$tf_stage_dir""/target_finder_model"
find "../target_finder_model/" -name "*.py" -exec cp "{}" \
  "$tf_stage_dir/target_finder_model/" \;

# Copy over configuration and informational files.
cp ../README.md ../LICENSE \
  ../setup.py "$tf_stage_dir"

# Compress the directory.
echo "Creating archive"

cd "../release/staging"
tar -czvf "$archive_name" "target-finder-model"
mv "$archive_name" ..

echo "\033[32mCreated target-finder-model release" \
  "(""$archive_name"")\033[0m"

# Remove the staging directory.
echo "Removing staging files"
cd ..
rm -rf staging
