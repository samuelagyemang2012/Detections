import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN, CSVLogger, LearningRateScheduler, ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from math import ceil
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.python.keras.callbacks import LearningRateScheduler
from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast
from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_L2Normalization import L2Normalization
from keras_loss_function.keras_ssd_loss import SSDLoss
from eval_utils.average_precision_evaluator import Evaluator
import misc_utils.inference_utils as msu
import pandas as pd

img_height = 448
img_width = 448
confidence_threshold = 0.5
iou_threshold = 0.5
# classes = ['background', 'car', 'bicyclist', 'pedestrian']
classes = ['background', 'car', 'bicyclist', 'pedestrian']
n_classes = len(classes)

model_path = "C:/Users/Administrator/Desktop/cruw_model_v2.h5"
base_path = "C:/Users/Administrator/Desktop/datasets/CRUW/ALL/resized/images/"
data_path = "data/cruw_test_data_rgb.csv"
# dest_path = "C:/Users/Administrator/Desktop/my_detections/cruw/single/"

df = pd.read_csv(data_path)
images_list = df['image'].unique()

# images_list = os.listdir(base_path)
K.clear_session()

model = msu.load_det_model(model_path)
detections = msu.do_detections(images_list,
                               base_path,
                               model,
                               img_height,
                               img_width,
                               confidence_threshold,
                               iou_threshold)

gts = msu.get_ground_truths(data_path, 'image')
processed_detections = msu.process_predictions(detections)

print(len(gts))
print(len(processed_detections))

msu.pickle_data(gts, "C:/Users/Administrator/Desktop/gts.sav")
msu.pickle_data(processed_detections, "C:/Users/Administrator/Desktop/single_preds.sav")


msu.get_mAP(processed_detections, gts, n_classes, iou_threshold, False)
print("Done")
