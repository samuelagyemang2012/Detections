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

K.clear_session()

# build ssd_300 model - for new mode
model = ssd_300(image_size=(img_height, img_width, img_channels),
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
weights_path = "C:/Users/Administrator/Desktop/datasets/trained_models/VGG_ILSVRC_16_layers_fc_reduced.h5"
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

# for pretrained model
# model_path = "./saved_models/final_model.h5"
# model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
#                                                'L2Normalization': L2Normalization,
#                                                'compute_loss': ssd_loss.compute_loss})

model.load_weights(weights_path, by_name=True)
adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

# Load data
train_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)
val_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)

# Data path
images_dir = 'C:/Users/Administrator/Desktop/datasets/CRUW/ALL/resized/CRUW_kaggle/images/'
train_labels_filename = 'data/cruw_train_data_rgb.csv'
val_labels_filename = 'data/cruw_val_data_rgb.csv'

# Load Data
train_dataset.parse_csv(images_dir=images_dir,
                        labels_filename=train_labels_filename,
                        # input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'],
                        input_format=['image_name', 'xmin', 'ymin', 'xmax', 'ymax', 'class_id'],
                        include_classes='all')

val_dataset.parse_csv(images_dir=images_dir,
                      labels_filename=val_labels_filename,
                      input_format=['image_name', 'xmin', 'ymin', 'xmax', 'ymax', 'class_id'],
                      include_classes='all')

# Data augmentation
batch_size = 4

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

train_generator = train_dataset.generate(batch_size=batch_size,
                                         shuffle=True,
                                         transformations=[ssd_data_augmentation],
                                         label_encoder=ssd_input_encoder,
                                         returns={'processed_images',
                                                  'encoded_labels'},
                                         keep_images_without_gt=False)

val_generator = val_dataset.generate(batch_size=batch_size,
                                     shuffle=False,
                                     transformations=[convert_to_3_channels, resize],
                                     label_encoder=ssd_input_encoder,
                                     returns={'processed_images',
                                              'encoded_labels'},
                                     keep_images_without_gt=False)

train_dataset_size = train_dataset.get_dataset_size()
val_dataset_size = val_dataset.get_dataset_size()

print("Number of images in the training dataset:\t{:>6}".format(train_dataset_size))
print("Number of images in the validation dataset:\t{:>6}".format(val_dataset_size))


def lr_schedule(epoch):
    if epoch < 80:
        return 0.001
    elif epoch < 100:
        return 0.0001
    else:
        return 0.00001


model_checkpoint = ModelCheckpoint(filepath='./weights/ssd_weights_v2.h5',
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=True,
                                   save_weights_only=False,
                                   mode='auto',
                                   period=1)

early_stopping = EarlyStopping(monitor='val_loss',
                               min_delta=0.0,
                               patience=5,
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

history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=steps_per_epoch,
                              epochs=final_epoch,
                              callbacks=callbacks,
                              validation_data=val_generator,
                              validation_steps=ceil(val_dataset_size / batch_size),
                              initial_epoch=initial_epoch)

# plt.figure(figsize=(20, 12))
# plt.plot(history.history['loss'], label='loss')
# plt.plot(history.history['val_loss'], label='val_loss')
# plt.legend(loc='upper right', prop={'size': 24})

model.save('./saved_models/cruw_model_v2.h5')
print("training complete. model saved")
# ##################################################################################################################

# Prediction
# predict_generator = val_dataset.generate(batch_size=2,
#                                          shuffle=True,
#                                          transformations=[convert_to_3_channels, resize],
#                                          label_encoder=None,
#                                          returns={'processed_images',
#                                                   'filenames',
#                                                   'inverse_transform',
#                                                   'original_images',
#                                                   'original_labels'},
#                                          keep_images_without_gt=False)
#
# batch_images, batch_filenames, batch_inverse_transforms, batch_original_images, batch_original_labels = next(
#     predict_generator)
#
# i = 0
#
# print("Image:", batch_filenames[i])
# print()
# print("Ground truth boxes:\n")
# print(np.array(batch_original_labels[i]))

# Make predictions
# y_pred = model.predict(batch_images)
# y_pred_decoded = decode_detections(y_pred,
#                                    confidence_thresh=0.5,
#                                    iou_threshold=0.45,
#                                    top_k=200,
#                                    normalize_coords=normalize_coords,
#                                    img_height=img_height,
#                                    img_width=img_width)
#
# y_pred_decoded_inv = apply_inverse_transforms(y_pred_decoded, batch_inverse_transforms)
#
# np.set_printoptions(precision=2, suppress=True, linewidth=90)
# print("Predicted boxes:\n")
# print('   class   conf xmin   ymin   xmax   ymax')
# print(y_pred_decoded_inv[i])
#
# # plt.figure(figsize=(20, 12))
# plt.imshow(batch_images[i])
#
# # Draw the predicted boxes onto the image
# colors = plt.cm.hsv(np.linspace(0, 1, n_classes + 1)).tolist()
# classes = ['background', 'car', 'cyclist', 'pedestrian']  # , 'bicyclist', 'light']
# # plt.figure(figsize=(20, 12))
# plt.imshow(batch_original_images[i])
#
# current_axis = plt.gca()
#
# for box in batch_original_labels[i]:
#     xmin = box[1]
#     ymin = box[2]
#     xmax = box[3]
#     ymax = box[4]
#     label = '{}'.format(classes[int(box[0])])
#     current_axis.add_patch(
#         plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color='green', fill=False, linewidth=2))
#     current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor': 'green', 'alpha': 1.0})
#
# for box in y_pred_decoded_inv[i]:
#     xmin = box[2]
#     ymin = box[3]
#     xmax = box[4]
#     ymax = box[5]
#     color = colors[int(box[0])]
#     label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
#     current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color=color, fill=False, linewidth=2))
#     current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor': color, 'alpha': 1.0})
#
# plt.savefig("./examples/detections.jpg", bbox_inches='tight')
