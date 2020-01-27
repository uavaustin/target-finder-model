import pathlib
import time

import glob
from PIL import Image

import target_finder_model as tfm


models = {
    'frcnn': tfm.inference.DetectionModel(),
    'clf': tfm.inference.ClfModel()
}

models['frcnn'].load()
models['clf'].load()

detector = models['frcnn']
clf = models['clf']

classes = tfm.OD_CLASSES


def prepare_data(imgs):

    dataset = {}
    for img in imgs:
        p = pathlib.Path(img)
        txt = img.replace("".join(p.suffixes), ".txt")

        class_ids = []
        with open(txt, 'r') as label_fn:
            for line in label_fn.readlines():
                # Get the class id
                class_id = tuple(float(val) for val in line.split(" "))[0]
                class_ids.append(classes[int(class_id)])
        
        dataset[img] = class_ids

    return dataset

def get_detections(dataset):

    results = {
        "TP": 0,
        "FP": 0,
        "FN": 0,
    }

    start = time.time()

    for idx, img in enumerate(dataset):

        if idx % 100 == 0:
            print(f"On image {idx} of {len(dataset)}")

        offset_dets = detector.predict([Image.open(img)])
        detections = []
        for obj in offset_dets[0]:
            detections.append(obj.class_name)
            # If detection in labels, TP
            if obj.class_name in dataset[img]:
                results["TP"] += 1
            else:
                results["FP"] += 1
                #print(f"Found {obj.class_name}, truth {dataset[img]} in {img}")
        
        for obj in dataset[img]:
            if obj not in detections:
                results["FN"] += 1
    
    total_t = time.time() - start
    
    print(f"{idx} images in {total_t} s. Average {idx/total_t} imgs/s")
    return results


def get_classifications(imgs):

    start = time.time()
    
    num_correct = 0
    for idx, img in enumerate(imgs):
        if idx % 100 == 0:
            print(f"On image {idx} of {len(imgs)}")

        # Get class id 
        has_target = 0 if "target" not in img else 1
        results = clf.predict([Image.open(img)])   

        if results[0].class_idx == has_target:
            num_correct += 1
            
    total_t = time.time() - start
    print(f"Classified {len(imgs)} images in {total_t} s. Average {len(imgs)/total_t} imgs/sec")

    return num_correct / len(imgs)


if __name__ == '__main__':


    img_dir = "scripts_generate/data/clf_val/images/*.png"
    imgs = glob.glob(img_dir)
    acc = get_classifications(imgs[0:2000])
    print(f"Classification accuracy: {acc}")

    img_dir = "scripts_generate/data/real/detector_val/images/*.png"
    imgs = glob.glob(img_dir)
    dataset = prepare_data(imgs[0:2000])

    results = get_detections(dataset)

    print(results)

    Recall = results["TP"] / (results["TP"] + results["FN"])

    print(f"Recall: {Recall}")

    Precision = results["TP"] / (results["TP"] + results["FP"])
    print(f"Precision: {Precision}")