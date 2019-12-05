"""
Script for freezing inception v3 model
TODO Adapt script for other classifiers
"""
import os
import glob
import tensorflow as tf 
import nets.nets_factory  #models/research/slim
import tensorflow.contrib.slim as slim
from preprocessing import inception_preprocessing

flags = tf.app.flags
tf.flags.DEFINE_string('ckpt_dir', '','Directory containing model ckpts')
tf.flags.DEFINE_string('output_path', 'models/clf/frozen_clf.pb', 'Output model path')
FLAGS = flags.FLAGS

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)


class NetDef(object):
    """Contains definition of a model
    name: Name of model
    url: (optional) Where to download archive containing checkpoint
    model_dir_in_archive: (optional) Subdirectory in archive containing
        checkpoint files.
    preprocess: Which preprocessing method to use for inputs.
    input_size: Input dimensions.
    slim: If True, use tensorflow/research/slim/nets to build graph. Else, use
        model_fn to build graph.
    postprocess: Postprocessing function on predictions.
    model_fn: Function to build graph if slim=False
    num_classes: Number of output classes in model. Background class will be
        automatically adjusted for if num_classes is 1001.
    """
    def __init__(self, name, url=None, model_dir_in_archive=None,
                checkpoint_name=None, preprocess='inception',
            input_size=299, slim=True, postprocess=tf.nn.softmax, model_fn=None, num_classes=2):
        self.name = name
        self.url = url
        self.model_dir_in_archive = model_dir_in_archive
        self.checkpoint_name = checkpoint_name
        if preprocess == 'inception':
            self.preprocess = inception_preprocessing.preprocess_image
        elif preprocess == 'vgg':
            self.preprocess = vgg_preprocessing.preprocess_image
        self.input_width = input_size
        self.input_height = input_size
        self.slim = slim
        self.postprocess = postprocess
        self.model_fn = model_fn
        self.num_classes = num_classes

    def get_input_dims(self):
        return self.input_width, self.input_height

    def get_num_classes(self):
        return self.num_classes

def get_checkpoint(model, model_dir):
    """Get the checkpoint. User may provide their own checkpoint via model_dir.
    If model_dir is None, attempts to download the checkpoint using url property
    from model definition (see get_netdef()). default_models_dir/model is first
    checked to see if the checkpoint was already downloaded. If not, the
    checkpoint will be downloaded from the url.
    model: string, the model name (see NETS table)
    model_dir: string, optional user provided checkpoint location
    default_models_dir: string, the directory where files are downloaded to
    returns: string, path to the checkpoint file containing trained model params
    """
    # User has provided a checkpoint
    if model_dir:
        checkpoint_path = find_checkpoint_in_dir(model_dir)
        if not checkpoint_path:
            print('No checkpoint was found in', model_dir)
            exit(1)
        return checkpoint_path

def find_checkpoint_in_dir(model_dir):
    # tf.train.latest_checkpoint will find checkpoints if a 'checkpoint' file is
    # present in the directory.
    checkpoint_path = tf.train.latest_checkpoint(model_dir)
    if checkpoint_path:
        return checkpoint_path

    # tf.train.latest_checkpoint did not find anything. Find .ckpt file
    # manually.
    files = glob.glob(os.path.join(model_dir, '*.ckpt*'))
    if len(files) == 0:
        return None
    # Use last file for consistency if more than one (may not actually be
    # "latest").
    checkpoint_path = sorted(files)[-1]
    # Trim after .ckpt-* segment. For example:
    # model.ckpt-257706.data-00000-of-00002 -> model.ckpt-257706
    parts = checkpoint_path.split('.')
    ckpt_index = [i for i in range(len(parts)) if 'ckpt' in parts[i]][0]
    checkpoint_path = '.'.join(parts[:ckpt_index+1])
    return checkpoint_path

def freeze_model(model, ckpt_dir, output_path):
    """Builds an image classification model by name
    This function builds an image classification model given a model
    name, parameter checkpoint file path, and number of classes.  This
    function performs some graph processing to produce a graph that is
    well optimized by the TensorRT package in TensorFlow 1.7+.
    model: string, the model name (see NETS table)
    model_dir: string, optional user provided checkpoint location
    default_models_dir: string, directory to store downloaded model checkpoints
    returns: tensorflow.GraphDef, the TensorRT compatible frozen graph
    """

    netdef = NetDef(name='inception_v3',input_size=299, num_classes=2)
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.per_process_gpu_memory_fraction=0.5

    with tf.Graph().as_default() as tf_graph:
        with tf.Session(config=tf_config) as tf_sess:
            tf_input = tf.placeholder(tf.float32, [None, netdef.input_height, netdef.input_width, 3], name='input')
            if netdef.slim:
                # TF Slim Model: get model function from nets_factory
                network_fn = nets.nets_factory.get_network_fn(netdef.name, netdef.num_classes,
                        is_training=False)
                tf_net, tf_end_points = network_fn(tf_input)
            else:
                # TF Official Model: get model function from NETS
                tf_net = netdef.model_fn(tf_input, training=False)

            tf_output = tf.identity(tf_net, name='logits')
            num_classes = tf_output.get_shape().as_list()[1]
            tf_output_classes = tf.argmax(tf_output, axis=1, name='classes')

            # Get checkpoint.
            checkpoint_path = get_checkpoint(model, ckpt_dir)
            print('Using checkpoint:', checkpoint_path)
            # load checkpoint
            tf_saver = tf.train.Saver()
            tf_saver.restore(save_path=checkpoint_path, sess=tf_sess)

            # freeze graph
            frozen_graph = tf.graph_util.convert_variables_to_constants(
                tf_sess,
                tf_sess.graph_def,
                output_node_names=['logits', 'classes']
            )
    with tf.io.gfile.GFile(output_path, "wb") as f:
        f.write(frozen_graph.SerializeToString())

    return 


if __name__ == '__main__':

    assert FLAGS.ckpt_dir

    if not tf.io.gfile.isdir(os.path.dirname(FLAGS.output_path)):
        tf.gfile.MakeDirs(os.path.dirname(FLAGS.output_path))

    freeze_model('inception_v3', FLAGS.ckpt_dir, FLAGS.output_path)