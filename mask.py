import sys
import os
import math
from operator import itemgetter
sys.path.append("player_recognition")

import player_recognition.model as modellib
import player_recognition.utils as pr_utils
import image_utils

from tqdm import tqdm
import coco
import numpy as np
from PIL import Image


ROOT_DIR = os.getcwd()
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "player_recognition/mask_rcnn_coco.h5")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

if not os.path.exists(COCO_MODEL_PATH):
    pr_utils.download_trained_weights(COCO_MODEL_PATH)


class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


class Mask(object):
    class_id = None
    score = None
    mask = None
    rois = None
    masked_image = None
    masked_image_np = None
    upper_half = None
    upper_half_np = None
    colour = None
    boundary_index = None
    center = None
    kmeans_label = None


    def __init__(self, class_id, mask, rois, score, masked_image):
        self.class_id = class_id
        self.mask = mask
        self.rois = rois
        self.center = (
            int((rois[3] + rois[1])/2),
            int((rois[2] + rois[0])/2)
        )
        self.score = score
        self.masked_image = masked_image
        self.masked_image_np = np.array(masked_image)
        self.xywh = image_utils.gen_xywh_from_box(rois)

        width, height = masked_image.size
        self.upper_half = masked_image.crop((0, 0, width, int(height/2)))
        self.upper_half_np = np.array(self.upper_half)

    def __repr__(self):
        return str({
            "class_id": self.class_id, 
            "score": self.score
            })


class Masks(list):

    def filter_boundary_index(self, index):
        result = []
        for mask in self:
            if mask.boundary_index == index:
                result.append(mask)
        return Masks(result)

    def filter_classify(self, classify):
        result = []
        for detection in self:
            if detection.classify == classify:
                result.append(detection)
        return Masks(result)

    def distances(self, MAX_DISTANCE=200):
        n = len(self)
        result = []

        for i in range(n):
            p1 = self[i].center
            for j in range(i+1, n):
                p2 = self[j].center
                distance = math.hypot(p2[0] - p1[0], 0.25*(p2[1] - p1[1]))
                result.append([i, j, distance])
        result = [val for val in result if val[2] <= MAX_DISTANCE]
        result.sort(key=itemgetter(2))
        return np.array(result)

    def get_xywh(self):
        return [s.xywh for s in self]


class MaskRCNN():
    def __init__(self):
        self.config = InferenceConfig()
        self.model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=self.config)
        self.model.load_weights(COCO_MODEL_PATH, by_name=True)

    def reinit(self):
        self.model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=self.config)
        self.model.load_weights(COCO_MODEL_PATH, by_name=True)

    def detect_people(self, image, threshold = 0.9):
        results = self.model.detect([image])
        r = results[0]

        self.config.BATCH_SIZE = 1
        self.config.IMAGES_PER_GPU = 1

        masks = []
        for i, class_id in enumerate(r["class_ids"]):
            if class_id != 1:
                continue
            crop = r["rois"][i] # top left bottom right
            score = r["scores"][i]
            this_mask = r["masks"][:, :, i:i+1]
            subimage = np.multiply(this_mask,image)

            mask = Mask(
                class_id,
                this_mask,
                crop,
                score,
                Image.fromarray(subimage).crop((crop[1], crop[0], crop[3], crop[2])) # left top right bottom
            )
            if mask.score > threshold:
                masks.append(mask)

        return Masks(masks)
