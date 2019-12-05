# Target Finder Model

> Related scripts and models for [target-finder](https://github.com/uavaustin/target-finder).

## Developer Instructions

### Install
1. Download `git clone https://github.com/uavaustin/target-finder-model && cd target-finder-model`
2. Install `pip install -r requirements-dev.txt` (requires CUDA installation)

### Usage

```python
import target_finder_model as tfm

model = tfm.inference.DetectionModel()
# or tfm.inference.ClfModel()
model.load()
objects = model.predict(['temp.jpg'])
```
#### Generate Data
* `python scripts_generate/pull_assets.py` Download base shapes and background images
* `python scripts_generate/create_detection_data.py` Create images for object detection training 
* `python scripts_generate/create_clf_data.py` Create images for classification training 
* `python scripts_tf/create_tf_records.py --image_dir ./scripts_generate/data --output_dir ./model_data` Reformat training files

### Training Pre-Classifier
```
python scrips_tf/train_clf.py
```
### Optimize Pre-Classifier from ckpt
```
python scripts_tf/optimize_clf.py \
    --model inception_v3 \
    --data_dir model_data/clf_records \
    --calib_data_dir model_data/clf_records \
    --model_path path/to/frozen_model.pb \
    --mode validation \
    --save_path path/to/optimized_clf_int8.pb \
    --use_trt \
    --precision INT8
```
### [Optional] Freeze Pre-Classifier 
```
python scripts_tf/freeze_clf.py --ckpt_dir=/folder/with/ckpts
```

### Training Object Detector Model

1. In a seperate folder `git clone https://github.com/tensorflow/models.git`
2. Run with `MODEL_NAME` set to one of the models in `models/2`
```
python path/to/models/research/object_detection/model_main.py \
    --pipeline_config_path=models/MODEL_NAME/pipeline.config \
    --model_dir=models/MODEL_NAME \
    --num_train_steps=5000 \
    --sample_1_of_n_eval_examples=1 \
    --alsologtostderr
```
#### Freeze Model Object Detector Model
This will create a `frozen_inference_graph.pb` that will be optimized with tensorrt.
```
python path/to/models/research/object_detection/export_inference_graph.py \
    --input_type=image_tensor \
    --pipeline_config_path=models/MODEL_NAME/pipeline.config \
    --trained_checkpoint_prefix=models/MODEL_NAME/model.ckpt \
    --output_directory=models/MODEL_NAME/frozen
```
### Optimize Object Detector Model
```
python scripts_tf/optimize.py --frozen_model=/path/to/frozen_inference_graph.pb --output_dir=/out/optimized_od.pb
```
## Repository Contents

#### Model Files
* `model_data/` Object detection training data, filled w/generate scripts
* `model/` Model configs, TensorFlow pipelines, training checkpoints

#### Scripts
* `scripts_generate/` Scripts for generating artifical training data
* `scripts_tf/` Helpful TensorFlow utilities

#### Misc
* `.circleci/` CI Config for automated builds and testing
* `Dockerfiles/` Docker resources for creating a prebuilt ML environment
* `target_finder_model/` The package that will be exported to [target-finder](https://github.com/uavaustin/target-finder) when a release is created.

## Testing
