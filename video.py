#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import sys
import os

import warnings
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO
from mask import MaskRCNN
from tqdm import tqdm

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import imageio
import image_utils
warnings.filterwarnings('ignore')

def get_filename(filename):
    split = os.path.splitext(filename)
    return split[0].split("/")[-1] + split[1]

def main(mask_rcnn):

    filename = sys.argv[1]
   # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0

   # deep_sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)


    reader = imageio.get_reader(filename, "ffmpeg")
    fps = reader.get_meta_data()['fps']
    N = len(reader) - 1

    writer = imageio.get_writer("output/" + get_filename(filename) , fps=fps)

    for i, frame in tqdm(enumerate(reader), desc="Frames ", total=N):

        masks = mask_rcnn.detect_people(frame)
        masks = image_utils.classify_masks(masks, by="average_colour"):
        boxs = masks.get_xywh()

        # print("box_num",len(boxs))
        features = encoder(frame, boxs)

        # score to 1.0 here).
        detections = [Detection(mask.get_xywh(), mask.score, feature, mask.kmeans_label) for mask, feature in zip(masks, features)]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores) #TODO: with maskrcnn, this may not be required
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)
       # _, masks = image_utils.apply_masks_to_image_np(frame, masks) #TODO: split into classify, then draw masks based on tracks

        image_utils.draw_player_with_tracks(frame, tracker.tracks):
        writer.append_data(frame)

    writer.close()

if __name__ == '__main__':
    main(MaskRCNN())
