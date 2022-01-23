import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, CSVLogger
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from math import ceil
import numpy as np
from matplotlib import pyplot as plt
from models.keras_ssd7 import ssd7
from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast
from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.data_augmentation_chain_constant_input_size import DataAugmentationConstantInputSize
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.object_detection_2d_geometric_ops import Resize
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)

# Hyper-parameters
img_height = 300
img_width = 300
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
subtract_mean = [123, 117, 104]
swap_channels = [2, 1, 0]

K.clear_session()

# build ssd_300 model
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
weights_path = "D:/Pycharm Projects/detection_models/VGG_ILSVRC_16_layers_fc_reduced.h5"
model.load_weights(weights_path, by_name=True)

adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

# Load data
train_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)
val_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)

# Data path
images_dir = 'D:/Datasets/my_coco/all/resized/images/'
train_labels_filename = 'D:/Datasets/my_coco/all/resized/detection_split/train.csv'
val_labels_filename = 'D:/Datasets/my_coco/all/resized/detection_split/val.csv'

# Load Data
train_dataset.parse_csv(images_dir=images_dir,
                        labels_filename=train_labels_filename,
                        input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'],
                        include_classes='all')

val_dataset.parse_csv(images_dir=images_dir,
                      labels_filename=val_labels_filename,
                      input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'],
                      include_classes='all')

train_dataset_size = train_dataset.get_dataset_size()
val_dataset_size = val_dataset.get_dataset_size()

print("Number of images in the training dataset:\t{:>6}".format(train_dataset_size))
print("Number of images in the validation dataset:\t{:>6}".format(val_dataset_size))

# Data augmentation
batch_size = 1

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
                                     shuffle=True,
                                     transformations=[convert_to_3_channels, resize],
                                     label_encoder=ssd_input_encoder,
                                     returns={'processed_images',
                                              'encoded_labels'},
                                     keep_images_without_gt=False)

# model_checkpoint = ModelCheckpoint(filepath='ssd7_weights.h5',
#                                    monitor='val_loss',
#                                    verbose=1,
#                                    save_best_only=True,
#                                    save_weights_only=False,
#                                    mode='auto',
#                                    period=1)

early_stopping = EarlyStopping(monitor='val_loss',
                               min_delta=0.0,
                               patience=10,
                               verbose=1)

reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss',
                                         factor=0.2,
                                         patience=8,
                                         verbose=1,
                                         min_delta=0.001,
                                         cooldown=0,
                                         min_lr=0.00001)

callbacks = [
    # model_checkpoint,
    early_stopping,
    reduce_learning_rate]


def train(model):
    initial_epoch = 0
    final_epoch = 20
    steps_per_epoch = 1000

    history = model.fit_generator(train_generator,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=final_epoch,
                                  callbacks=callbacks,
                                  validation_data=val_generator,
                                  validation_steps=ceil(val_dataset_size / batch_size),
                                  initial_epoch=initial_epoch)

    plt.figure(figsize=(20, 12))
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend(loc='upper right', prop={'size': 24})
    model.save('model.h5')
    return model


model = train(model)

# ##################################################################################################################

# Prediction
predict_generator = val_dataset.generate(batch_size=6,
                                         shuffle=True,
                                         transformations=[],
                                         label_encoder=None,
                                         returns={'processed_images',
                                                  'processed_labels',
                                                  'filenames'},
                                         keep_images_without_gt=False)

batch_images, batch_labels, batch_filenames = next(predict_generator)

i = 0

print("Image:", batch_filenames[i])
print()
print("Ground truth boxes:\n")
print(batch_labels[i])
print(len(batch_filenames))

y_pred = model.predict(batch_images)

# 4: Decode the raw prediction `y_pred`

y_pred_decoded = decode_detections(y_pred * 266.,
                                   confidence_thresh=0.5,
                                   iou_threshold=0.45,
                                   top_k=200,
                                   normalize_coords=normalize_coords,
                                   img_height=img_height,
                                   img_width=img_width)

np.set_printoptions(precision=2, suppress=True, linewidth=90)
print("Predicted boxes:\n")
print('   class   conf xmin   ymin   xmax   ymax')
print(y_pred_decoded[i])

plt.figure(figsize=(20, 12))
plt.imshow(batch_images[i])

current_axis = plt.gca()

print("prediction")
colors = plt.cm.hsv(np.linspace(0, 1, n_classes + 1)).tolist()
classes = ['background', 'bird', 'dog', 'car']

# for box in batch_labels[i]:
#     xmin = box[1]
#     ymin = box[2]
#     xmax = box[3]
#     ymax = box[4]
#     label = '{}'.format(classes[int(box[0])])
#     current_axis.add_patch(
#         plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color='green', fill=False, linewidth=2))
#     current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor': 'green', 'alpha': 1.0})

for box in y_pred_decoded[i]:
    xmin = box[-4]
    ymin = box[-3]
    xmax = box[-2]
    ymax = box[-1]

    print('predicted:')
    print(xmin, ymin, xmax, ymax)
    color = colors[int(box[0])]
    label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
    current_axis.add_patch(
        plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color="#0099FF", fill=False, linewidth=2))

plt.savefig("detections.jpg", bbox_inches='tight')
