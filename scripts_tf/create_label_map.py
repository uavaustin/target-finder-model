"""
Create label map file.
"""
import os

with open(os.path.join(os.path.dirname(__file__),
          os.pardir, 'config.yaml'), 'r') as stream:
    import yaml
    config = yaml.safe_load(stream)

"""
CLASSES = []
for shape in config['classes']['shapes']:
    for alpha in config['classes']['alphas']:
        CLASSES.append('-'.join([shape, alpha]))
"""
CLASSES = []
for shape in config['classes']['shapes']:
    CLASSES.append(shape)
for alpha in config['classes']['alphas']:
    CLASSES.append(alpha)
"""
for alpha in config['classes']['alphas']:
    CLASSES.append(alpha)
"""

with open(os.path.join(os.path.dirname(__file__),
          os.pardir, 'model_data', 'shape_label_map.pbtxt'), 'w') as fp:
    for i, name in enumerate(CLASSES):
        fp.write("item {{\n id: {}\n name: '{}'\n}}\n\n".format(i + 1, name))
