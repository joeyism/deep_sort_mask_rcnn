#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import sys
import os
import random

import warnings
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO
from mask import MaskRCNN
from tqdm import tqdm
import imageio

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import image_utils
import tracker_utils
warnings.filterwarnings('ignore')

def get_filename(filename):
    split = os.path.splitext(filename)
    return split[0].split("/")[-1] + split[1]

def detect(frame, tracker, encoder, mask_rcnn, nms_max_overlap=1.0, force_draw=False, threshold=0.9):
    masks = mask_rcnn.detect_people(frame, threshold=threshold)

    # skip if no of player si sthe same
    #if len(tracker.tracks) != len(masks):
    #    masks = image_utils.classify_masks_with_hash(masks)

    masks = image_utils.classify_masks_with_hash(masks)
    boxs = masks.get_xywh()


    features = encoder(frame, boxs)


    detections = [Detection(mask.xywh, mask.score, feature, mask.kmeans_label) for mask, feature in zip(masks, features)]

    # Run non-maxima suppression.
    boxes = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores) #TODO: with maskrcnn, this may not be required
    detections = [detections[i] for i in indices]

    # Call the tracker
    tracker.predict()
    tracker.update(detections)
    if random.randint(0, 20) == 1:
        tracker_utils.remove_deleted_tracks(tracker)

    image_utils.draw_player_with_tracks(frame, tracker.tracks, force=force_draw)
    return frame, tracker, encoder


def main(mask_rcnn):
    filename = sys.argv[1]
    image = image_utils.load_image_into_numpy_array(Image.open(filename))
   # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None

   # deep_sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    image, _, _ = detect(image.copy(), tracker, encoder, mask_rcnn, force_draw=True)
    Image.fromarray(image).save("output/" + get_filename(filename))

if __name__ == '__main__':
    main(MaskRCNN())
