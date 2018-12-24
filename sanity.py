#TODO: delete this file when done
import sys
sys.path.append("player_recognition")

import numpy as np

from yolo import YOLO
from PIL import Image
from mask import MaskRCNN

yolo = YOLO()
yolo_boxes = yolo.detect_image(Image.open("player_recognition/sports_images/liverpool-chelsea.jpeg"))

mask_rcnn = MaskRCNN()
mask_boxes = mask_rcnn.model.detect([np.asarray(Image.open("player_recognition/sports_images/liverpool-chelsea.jpeg"))])
