"""
Create label map file.
"""
import os


CLASSES = (
    'circle,cross,pentagon,quarter-circle,rectangle,semicircle,square,star,'
    'trapezoid,triangle'
).split(',') + list('ABCDEFGHIJKLMNOPQRSTUVWXYZ4')


with open('shape_label_map.pbtxt', 'w') as fp:
    for i, name in enumerate(CLASSES):
        fp.write("item {{\n id: {}\n name: {}\n}}\n\n".format(i + 1, name))