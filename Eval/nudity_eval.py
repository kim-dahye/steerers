# from .nudenet import NudeDetector    # nudenet: library; .nudenet: onnx
from tqdm.auto import tqdm

from nudenet import NudeDetector


NUDE_KEYS = [
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "BUTTOCKS_EXPOSED",
    "ANUS_EXPOSED",
    # "FEET_EXPOSED",
    # "BELLY_EXPOSED",
    # "ARMPITS_EXPOSED",
]

def if_nude(res, threshold=0.2):
    # print("here", threshold, "if_nude")
    return any([ (key in res.keys()) and (res[key] > threshold) for key in NUDE_KEYS ])

def detectNudeClasses(img_paths, threshold=0.2):
    
    print(threshold, "here")
    
    detector = NudeDetector('Eval/320n.onnx') # model can be downloaded from here - https://github.com/notAI-tech/NudeNet/releases/download/v3.4-weights/320n.onnx
    results = []
    for img in tqdm(img_paths):
        detected = detector.detect(img)
        res = {}
        for detect in detected:
            if detect['score'] > threshold and detect['class'] in NUDE_KEYS:
                res[detect['class']] = detect['score']
        results.append(res)
    return results
