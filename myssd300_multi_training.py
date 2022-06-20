import cv2
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
from models.multi_keras_ssd300 import multi_ssd_300
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
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)

# Hyper-parameters
img_height = 448
img_width = 448
img_channels = 3
n_classes = 3
mode = 'training'
l2_regularization = 0.0005
scales = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05]
aspect_ratios_per_layer = [[1.0, 2.0, 0.5],
                           [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                           [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                           [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                           [1.0, 2.0, 0.5],
                           [1.0, 2.0, 0.5]]
two_boxes_for_ar1 = True
steps = [8, 16, 32, 64, 100, 300]
offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
clip_boxes = False
variances = [0.1, 0.1, 0.2, 0.2]
normalize_coords = True
subtract_mean = [0, 0, 0]  # [123, 117, 104]  # [0, 0, 0]
swap_channels = [0, 1, 2]  # [2, 1, 0]
batch_size = 4
K.clear_session()

# build ssd_300 model - for new mode
model = multi_ssd_300(image_size=(img_height, img_width, img_channels),
                      n_classes=n_classes,
                      mode='training',
                      l2_regularization=l2_regularization,
                      scales=scales,
                      aspect_ratios_per_layer=aspect_ratios_per_layer,
                      two_boxes_for_ar1=two_boxes_for_ar1,
                      steps=steps,
                      offsets=offsets,
                      clip_boxes=clip_boxes,
                      variances=variances,
                      normalize_coords=normalize_coords,
                      subtract_mean=subtract_mean,
                      swap_channels=swap_channels)

# Load weights: optional
# weights_path = "C:/Users/Administrator/Desktop/datasets/trained_models/VGG_ILSVRC_16_layers_fc_reduced.h5"
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

# for pretrained model
# model_path = "./saved_models/final_model.h5"
# model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
#                                                'L2Normalization': L2Normalization,
#                                                'compute_loss': ssd_loss.compute_loss})

# model.load_weights(weights_path, by_name=True)
adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

# Load data
rgb_train_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)
rgb_val_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)

rf_train_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)
rf_val_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)

# Data path
rgb_images_dir = 'C:/Users/Administrator/Desktop/datasets/CRUW/ALL/resized/CRUW_kaggle/images/'
rf_images_dir = 'C:/Users/Administrator/Desktop/datasets/CRUW/ALL/resized/CRUW_kaggle/rf_maps/'

rgb_train_labels_filename = 'data/cruw_train_rgb.csv'
rf_train_labels_filename = 'data/cruw_train_rf.csv'

rgb_val_labels_filename = 'data/cruw_valid_rgb.csv'
rf_val_labels_filename = 'data/cruw_valid_rf.csv'

# Load Data
rgb_train_dataset.parse_csv(images_dir=rgb_images_dir,
                            labels_filename=rgb_train_labels_filename,
                            # input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'],
                            input_format=['image_name', 'xmin', 'ymin', 'xmax', 'ymax', 'class_id'],
                            include_classes='all')

rgb_val_dataset.parse_csv(images_dir=rgb_images_dir,
                          labels_filename=rgb_val_labels_filename,
                          input_format=['image_name', 'xmin', 'ymin', 'xmax', 'ymax', 'class_id'],
                          include_classes='all')

rf_train_dataset.parse_csv(images_dir=rf_images_dir,
                           labels_filename=rf_train_labels_filename,
                           # input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'],
                           input_format=['image_name', 'xmin', 'ymin', 'xmax', 'ymax', 'class_id'],
                           include_classes='all')

rf_val_dataset.parse_csv(images_dir=rf_images_dir,
                         labels_filename=rf_val_labels_filename,
                         input_format=['image_name', 'xmin', 'ymin', 'xmax', 'ymax', 'class_id'],
                         include_classes='all')

# Data augmentation


# data_augmentation_chain = DataAugmentationConstantInputSize(random_brightness=(-48, 48, 0.5),
#                                                             random_contrast=(0.5, 1.8, 0.5),
#                                                             random_saturation=(0.5, 1.8, 0.5),
#                                                             random_hue=(18, 0.5),
#                                                             random_flip=0.5,
#                                                             random_translate=((0.03, 0.5), (0.03, 0.5), 0.5),
#                                                             random_scale=(0.5, 2.0, 0.5),
#                                                             n_trials_max=3,
#                                                             clip_boxes=True,
#                                                             overlap_criterion='area',
#                                                             bounds_box_filter=(0.3, 1.0),
#                                                             bounds_validator=(0.5, 1.0),
#                                                             n_boxes_min=1,
#                                                             background=(0, 0, 0))

ssd_data_augmentation = SSDDataAugmentation(img_height=img_height,
                                            img_width=img_width,
                                            background=subtract_mean)

convert_to_3_channels = ConvertTo3Channels()
resize = Resize(height=img_height, width=img_width)

# predictor_sizes = [model.get_layer('classes4').output_shape[1:3],
#                    model.get_layer('classes5').output_shape[1:3],
#                    model.get_layer('classes6').output_shape[1:3],
#                    model.get_layer('classes7').output_shape[1:3]]

predictor_sizes = [model.get_layer('conv4_3_norm_mbox_conf').output_shape[1:3],
                   model.get_layer('fc7_mbox_conf').output_shape[1:3],
                   model.get_layer('conv6_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv7_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv8_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv9_2_mbox_conf').output_shape[1:3]]

ssd_input_encoder = SSDInputEncoder(img_height=img_height,
                                    img_width=img_width,
                                    n_classes=n_classes,
                                    predictor_sizes=predictor_sizes,
                                    scales=scales,
                                    aspect_ratios_per_layer=aspect_ratios_per_layer,
                                    two_boxes_for_ar1=two_boxes_for_ar1,
                                    steps=steps,
                                    offsets=offsets,
                                    clip_boxes=clip_boxes,
                                    variances=variances,
                                    matching_type='multi',
                                    pos_iou_threshold=0.5,
                                    neg_iou_limit=0.5,
                                    normalize_coords=normalize_coords)


def multi_gen(rgb_gen, rf_gen, batch_size_, ssd_data_augmentation_, ssd_input_encoder_):
    rgb = rgb_gen.generate(batch_size=batch_size_,
                           shuffle=True,
                           transformations=[],  # [ssd_data_augmentation_],
                           label_encoder=ssd_input_encoder_,
                           returns={'processed_images',
                                    'encoded_labels'},
                           keep_images_without_gt=False)

    rf = rf_gen.generate(batch_size=batch_size_,
                         shuffle=True,
                         transformations=[],  # [ssd_data_augmentation_],
                         label_encoder=ssd_input_encoder_,
                         returns={'processed_images',
                                  'encoded_labels'},
                         keep_images_without_gt=False)

    while True:
        rgb_X = next(rgb)
        rf_X = next(rf)
        yield [rgb_X[0], rf_X[0]], rgb_X[1]


train_multi_gen = multi_gen(rgb_train_dataset, rf_train_dataset, batch_size, ssd_data_augmentation, ssd_input_encoder)
val_multi_gen = multi_gen(rgb_val_dataset, rf_val_dataset, batch_size, ssd_data_augmentation, ssd_input_encoder)

t_imgs, t_labels = next(train_multi_gen)
v_imgs, v_labels = next(val_multi_gen)


# for t in t_imgs:
#     for i in range(0, 2):
#         cv2.imshow("", t[i])
#         cv2.waitKey(-1)

def lr_schedule(epoch):
    if epoch < 80:
        return 0.001
    elif epoch < 100:
        return 0.0001
    else:
        return 0.00001


model_checkpoint = ModelCheckpoint(filepath='./weights/ssd_weights_multi.h5',
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=True,
                                   save_weights_only=False,
                                   mode='auto',
                                   period=1)

early_stopping = EarlyStopping(monitor='val_loss',
                               min_delta=0.0,
                               patience=10,
                               verbose=1)

learning_rate_scheduler = LearningRateScheduler(schedule=lr_schedule,
                                                verbose=1)

terminate_on_nan = TerminateOnNaN()

# reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss',
#                                          factor=0.2,
#                                          patience=8,
#                                          verbose=1,
#                                          min_delta=0.001,
#                                          cooldown=0,
#                                          min_lr=0.00001)

callbacks = [
    model_checkpoint,
    early_stopping,
    terminate_on_nan,
    learning_rate_scheduler]

# Train
initial_epoch = 0
final_epoch = 150  # 100
steps_per_epoch = 1000

history = model.fit(train_multi_gen,
                    steps_per_epoch=steps_per_epoch,
                    epochs=final_epoch,
                    callbacks=callbacks,
                    validation_data=val_multi_gen,
                    validation_steps=ceil(724 / batch_size),
                    initial_epoch=initial_epoch)

# plt.figure(figsize=(20, 12))
# plt.plot(history.history['loss'], label='loss')
# plt.plot(history.history['val_loss'], label='val_loss')
# plt.legend(loc='upper right', prop={'size': 24})

model.save('./saved_models/cruw_model_multi.h5')
print("training complete. model saved")
