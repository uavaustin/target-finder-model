"""
Create label map file.
"""
import os

with open(os.path.join(os.path.dirname(__file__), 
        os.pardir, 'config.yaml'), 'r') as stream:
    import yaml
    config = yaml.safe_load(stream)

CLASSES = config['classes']['shapes'] + config['classes']['alphas']


with open(os.path.join(os.path.dirname(__file__), 
        os.pardir, 'model_data', 'shape_label_map.pbtxt'), 'w') as fp:
    for i, name in enumerate(CLASSES):
        fp.write("item {{\n id: {}\n name: '{}'\n}}\n\n".format(i + 1, name))
