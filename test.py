# import tensorflow as tf
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN, CSVLogger, LearningRateScheduler, ModelCheckpoint
# from tensorflow.keras import backend as K
# from tensorflow.keras.models import load_model
# from math import ceil
# import numpy as np
# from matplotlib import pyplot as plt
# from tensorflow.python.keras.callbacks import LearningRateScheduler
# from models.keras_ssd300 import ssd_300
# from keras_loss_function.keras_ssd_loss import SSDLoss
# from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
# from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast
# from data_generator.object_detection_2d_data_generator import DataGenerator
# from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation
# from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
# from data_generator.object_detection_2d_geometric_ops import Resize
# from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms
# from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
# from keras_layers.keras_layer_L2Normalization import L2Normalization
# from keras_loss_function.keras_ssd_loss import SSDLoss
# import os
#
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
# sess = tf.compat.v1.Session(config=config)
#
# # Hyper-parameters
# img_height = 300
# img_width = 480
# img_channels = 3
# n_classes = 5
# mode = 'training'
# l2_regularization = 0.0005
# scales = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05]
# aspect_ratios_per_layer = [[1.0, 2.0, 0.5],
#                            [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
#                            [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
#                            [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
#                            [1.0, 2.0, 0.5],
#                            [1.0, 2.0, 0.5]]
# two_boxes_for_ar1 = True
# steps = [8, 16, 32, 64, 100, 300]
# offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
# clip_boxes = False
# variances = [0.1, 0.1, 0.2, 0.2]
# normalize_coords = True
# subtract_mean = [123, 117, 104]  # [0, 0, 0]
# swap_channels = [2, 1, 0]
#
# K.clear_session()
#
# # build ssd_300 model
# model = ssd_300(image_size=(img_height, img_width, img_channels),
#                 n_classes=n_classes,
#                 mode='training',
#                 l2_regularization=l2_regularization,
#                 scales=scales,
#                 aspect_ratios_per_layer=aspect_ratios_per_layer,
#                 two_boxes_for_ar1=two_boxes_for_ar1,
#                 steps=steps,
#                 offsets=offsets,
#                 clip_boxes=clip_boxes,
#                 variances=variances,
#                 normalize_coords=normalize_coords,
#                 subtract_mean=subtract_mean,
#                 swap_channels=swap_channels)
#
# # # Load weights: optional
# # weights_path = "C:/Users/Administrator/Desktop/datasets/trained_models/VGG_ILSVRC_16_layers_fc_reduced.h5"
#
# ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
# adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
# model.compile(optimizer=adam, loss=ssd_loss.compute_loss)
# model.load_weights('D:/Datasets/Self Driving Cars/final_model.h5')
# model.save('C:/Users/Sam/Desktop/xxx.h5')

import os
import numpy as np
from tensorflow.keras import backend as K
import misc_utils.inference_utils as msu
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2
from ssd_encoder_decoder.ssd_output_decoder import decode_detections

img_height = 300
img_width = 480
confidence_threshold = 0.4
classes = ['background', 'car', 'truck', 'pedestrian', 'bicyclist', 'light']

model_path = "C:/Users/Sam/Desktop/xxx.h5"
base_path = "D:/Datasets/subsets/"  # "C:/Users/Administrator/Desktop/Self Driving Cars/images/"
dest_path = "C:/Users/Sam/Desktop/detections/ssd/"

images_list = os.listdir(base_path)
K.clear_session()

# model = msu.load_det_model(model_path)

# cc = []
#
# for i in images_list:
#     # img_ = image.load_img(base_path + i, target_size=(img_height, img_width))
#     # img_ = image.img_to_array(img_)
#
#     # img_1 = cv2.imread(base_path + images_list[0])
#     # img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
#     # img_1 = Image.fromarray(img_1)
#     # img_1 = img_1.resize((img_height, img_width))
#     # img_1 = image.img_to_array(img_1)
#     # image_array_expanded = np.expand_dims(img_1, axis=0)
#
#     cc.append(img_1)
#
# # print(preds)
# # image_ = msu.cv_show_detection(preds, image_, (0, 0, 255), 1, [])
# # cv2.imshow("", image_)
# # cv2.waitKey(0)/
#
# cc = np.array(cc).astype('float')
# cc = cc.reshape([len(cc), img_height, img_width, 3])
# print(cc.shape)
# preds1 = model.predict(cc)
# preds1 = decode_detections(preds1,
#                            confidence_thresh=0.5,
#                            iou_threshold=0.5,
#                            top_k=200,
#                            normalize_coords=True,
#                            img_height=img_height,
#                            img_width=img_width)
# print(preds1)

#
# print("cv2")
# img_1 = cv2.imread(base_path + images_list[0])
# img_1 = cv2.resize(img_1, (img_height, img_width), interpolation=cv2.INTER_AREA)
# img_1 = (img_1[..., ::-1].astype(np.float32))
# print(img_1)
# print("")
# print("----------------------------------------------------------------------")
# print("tf")
# img_2 = image.load_img(base_path + images_list[0], target_size=(img_height, img_width))
# img_2 = image.img_to_array(img_2)
# print(img_2)
# print("")

import cv2

cap = cv2.VideoCapture("C:/Users/Sam/Desktop/video.mp4")

# Check if camera opened successfully
if not cap.isOpened():
    print("Error opening video stream or file")

c = 0
# Read until video is completed
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:
        c += 1
        # Display the resulting frame
        cv2.imshow('Frame', frame)
        print(c)
        cv2.imwrite("C:/Users/Sam/Desktop/dd/" + str(c) + ".png",frame)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
