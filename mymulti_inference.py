import cv2
import pandas as pd
import misc_utils.inference_utils as msu
from tensorflow.keras.preprocessing import image
import numpy as np


classes = ['background', 'car', 'bicyclist', 'pedestrian']
multi_model_path = "C:/Users/Administrator/Desktop/cruw_model_multi_kaggle.h5"
rgb_img_base_path = "C:/Users/Administrator/Desktop/datasets/CRUW/ALL/resized/CRUW_kaggle/images/"
rf_img_base_path = "C:/Users/Administrator/Desktop/datasets/CRUW/ALL/resized/CRUW_kaggle/rf_maps/"
dest_path = "C:/Users/Administrator/Desktop/my_detections/cruw/multi/kaggle/"
rgb_data_path = "data/cruw_test_rgb.csv"
rf_data_path = "data/cruw_test_rf.csv"

rgb_image_path = "e0000000028.jpg"
rf_image_path = "e000028_0000.npy.png"

img_height_ = 448
img_width_ = 448

multi_data = []
rgb_data = []

# def load_det_model(model_path_):
#     ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)
#     model = load_model(model_path_, custom_objects={'AnchorBoxes': AnchorBoxes,
#                                                     'L2Normalization': L2Normalization,
#                                                     'DecodeDetections': DecodeDetections,
#                                                     'compute_loss': ssd_loss.compute_loss})
#     return model
#
#
# def do_detections(images_list_, image_base_path_, model_, img_height, img_width):
#     orig_images = []  # Store the images here.
#     resize_images = []
#
#     for i in images_list_:
#         img_ = imread(image_base_path_ + i)
#         orig_images.append([img_.shape[0], img_.shape[1]])
#         img = image.load_img(image_base_path_ + i, target_size=(img_height, img_width))
#         img = image.img_to_array(img)
#         resize_images.append(img)
#
#     resize_images = np.array(resize_images)
#
#     # Do predictions
#     preds = model_.predict(resize_images)
#     # Decode predictions
#     preds_decoded = decode_detections(preds,
#                                       confidence_thresh=0.5,
#                                       iou_threshold=0.5,
#                                       top_k=200,
#                                       normalize_coords=True,
#                                       img_height=img_height,
#                                       img_width=img_width)
#
#     return preds_decoded, orig_images
#
#
# def print_detections(detections_):
#     for d in detections_:
#         np.set_printoptions(precision=2, suppress=True, linewidth=90)
#         print("Predicted boxes:\n")
#         print('   class   conf xmin   ymin   xmax   ymax')
#         print(d)
#
#
# def show_detections(detections_, image_list, image_base, class_names, dest_path_, img_height, img_width):
#     for i, d in enumerate(detections_):
#         plt.figure(figsize=(20, 12))
#         current_axis = plt.gca()
#         plt.axis(False)
#         image_ = imread(image_base + image_list[i])
#         name = image_list[i]
#         plt.imshow(image_)
#         for box in d:
#             xmin = box[2] * image_.shape[1] / img_width
#             ymin = box[3] * image_.shape[0] / img_height
#             xmax = box[4] * image_.shape[1] / img_width
#             ymax = box[5] * image_.shape[0] / img_height
#
#             label = '{}: {:.2f}'.format(class_names[int(box[0])], box[1])
#             current_axis.add_patch(
#                 plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color="orange", fill=False, linewidth=1))
#             current_axis.text(xmin, ymin, label, size='x-large', color='white',
#                               bbox={'facecolor': "orange", 'alpha': 1.0})
#
#         plt.savefig(dest_path_ + name, dpi=100, bbox_inches="tight")


# load model
multi_model = msu.load_det_model(multi_model_path)

# select data
rgb_df = pd.read_csv(rgb_data_path)
rf_df = pd.read_csv(rf_data_path)

rgb_list = rgb_df['image'].to_list()
rgb_list = rgb_list[0:40]

rf_list = rf_df['image'].to_list()
rf_list = rf_list[0:40]

# load rgb image
for i in range(0, len(rgb_list)):
    rgb_img = image.load_img(rgb_img_base_path + rgb_list[i])
    rgb_img = image.img_to_array(rgb_img)
    rgb_img = np.array([rgb_img])

    # load rf map
    rf_img = img = image.load_img(rf_img_base_path + rf_list[i])
    rf_img = image.img_to_array(rf_img)
    rf_img = np.array([rf_img])

    multi_data.append([rgb_img, rf_img])

for i, m in enumerate(multi_data):
    # predict
    pred = multi_model.predict(m)

    # decode detections
    decoded_preds = msu.decode_detections(pred,
                                          confidence_thresh=0.5,
                                          iou_threshold=0.5,
                                          top_k=200,
                                          normalize_coords=True,
                                          img_height=img_height_,
                                          img_width=img_width_)
    # show detections
    msu.print_detections(decoded_preds)

    # show detections
    msu.show_detections(decoded_preds, [rgb_list[i]], rgb_img_base_path, classes, dest_path, img_height_, img_width_)
