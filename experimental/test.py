import target_finder_model as tfm
import time
import glob
import tensorflow.python.compiler.tensorrt

#Test OD 
img_list = glob.glob('test/od/*.png')
model = tfm.inference.DetectionModel()
model.load()
for i in range(0,10):
    t1 = time.time()
    objects = model.predict(img_list)
    print(time.time() - t1)

#Test Clf 
img_list = glob.glob('test/clf/*.png')
model = tfm.inference.ClfModel()
model.load()
for i in range(0,100):
    t1 = time.time()
    objects = model.predict(img_list)
    print(time.time() - t1)
