# https://github.com/tensorflow/tensorrt/tree/master/tftrt/examples/object_detection

from tftrt.examples.object_detection import (
    download_model, optimize_model,
    download_dataset, benchmark_model
)

frozen_graph = build_model(
    model_name="ssd_resnet_50_fpn_coco",
    input_dir="/models/object_detection/combined_nms_enabled",
    batch_size=8,
    override_nms_score_threshold=0.3,
)

frozen_graph = optimize_model(
    frozen_graph,
    use_trt=True,
    precision_mode="INT8",
    calib_images_dir="/data/coco-2017/train2017",
    num_calib_images=8,
    calib_batch_size=8,
    calib_image_shape=[640, 640],
    max_workspace_size_bytes=17179869184,
)

images_dir, annotation_path = download_dataset('val2014', output_dir='dataset')

statistics = benchmark_model(
    frozen_graph=frozen_graph, 
    images_dir="/data/coco2017/val2017",
    annotation_path="/data/coco2017/annotations/instances_val2017.json",
    batch_size=8,
    image_shape=[640, 640],
    num_images=4096,
    output_path="stats/ssd_resnet_50_fpn_coco_trt_int8.json"
)
