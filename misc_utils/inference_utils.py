import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from imageio import imread
import numpy as np
from matplotlib import pyplot as plt
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_L2Normalization import L2Normalization
from keras_loss_function.keras_ssd_loss import SSDLoss
from ssd_encoder_decoder.ssd_output_decoder import decode_detections
import pandas as pd
from mean_average_precision import MetricBuilder
import pickle
from tqdm import tqdm


def load_det_model(model_path_):
    ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)
    model = load_model(model_path_, custom_objects={'AnchorBoxes': AnchorBoxes,
                                                    'L2Normalization': L2Normalization,
                                                    'DecodeDetections': DecodeDetections,
                                                    'compute_loss': ssd_loss.compute_loss})
    return model


def do_detections(images_list_, image_base_path_, model_, img_height, img_width, confidence_threshold, iou_threshold):
    orig_images = []  # Store the images here.
    resize_images = []

    for i in images_list_:
        img_ = imread(image_base_path_ + i)
        orig_images.append([img_.shape[0], img_.shape[1]])
        img = image.load_img(image_base_path_ + i, target_size=(img_height, img_width))
        img = image.img_to_array(img)
        resize_images.append(img)

    resize_images = np.array(resize_images)

    # Do predictions
    preds = model_.predict(resize_images)
    # Decode predictions
    preds_decoded = decode_detections(preds,
                                      confidence_thresh=confidence_threshold,
                                      iou_threshold=iou_threshold,
                                      top_k=200,
                                      normalize_coords=True,
                                      img_height=img_height,
                                      img_width=img_width)

    return preds_decoded


def do_multi_detections(rgb_list_, rgb_base_path_, rf_list_, rf_base_path_, model_, img_height, img_width,
                        confidence_threshold, iou_threshold):
    multi_data = []
    preds = []

    for i in range(0, len(rgb_list_)):
        rgb_img = image.load_img(rgb_base_path_ + rgb_list_[i])
        rgb_img = image.img_to_array(rgb_img)
        rgb_img = np.array([rgb_img])

        # load rf map
        rf_img = image.load_img(rf_base_path_ + rf_list_[i])
        rf_img = image.img_to_array(rf_img)
        rf_img = np.array([rf_img])

        multi_data.append([rgb_img, rf_img])

    for i, m in enumerate(multi_data):
        # predict
        pred = model_.predict(m)

        # decode detections
        decoded_preds = decode_detections(pred,
                                          confidence_thresh=confidence_threshold,
                                          iou_threshold=iou_threshold,
                                          top_k=200,
                                          normalize_coords=True,
                                          img_height=img_height,
                                          img_width=img_width)
        for dp in decoded_preds:
            preds.append(dp)

    return np.array(preds)


def print_detections(detections_):
    for d in detections_:
        np.set_printoptions(precision=2, suppress=True, linewidth=90)
        print("Predicted boxes:\n")
        print('   class   conf xmin   ymin   xmax   ymax')
        print(d)


# def cv_show_detections(detections_, image_list, image_base, class_names, dest_path_):
#     for i, d in enumerate(detections_):
#         image_ = cv2.imread(image_base + image_list[i])
#         name = image_list[i]
#         for box in d:
#             xmin = box[2]
#             ymin = box[3]
#             xmax = box[4]
#             ymax = box[5]
#
#             image_ = cv2.rectangle(image_, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
#
#
# def cv_show_detection(detections_, image_, color, linewidth, class_names):
#     for detection in detections_:
#         for box in detection:
#             print(box)
#             xmin = int(box[2])
#             ymin = int(box[3])
#             xmax = int(box[4])
#             ymax = int(box[5])
#
#             image_ = cv2.rectangle(image_, (xmin, ymin), (xmax, ymax), color, linewidth)
#
#     return image_

def cv2_draw_box_with_labels(img_array, xmin, ymin, xmax, ymax, class_, conf, bb_color, line_width):
    label = class_ + " " + str(conf)
    img = cv2.rectangle(img_array, (xmin, ymin), (xmax, ymax), bb_color, line_width)

    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 0.6, 1)

    img = cv2.rectangle(img, (xmin, ymin - 15), (xmin + w, ymin), bb_color, -1)

    img = cv2.putText(img, label, (xmin, ymin - 5), cv2.FONT_HERSHEY_PLAIN, 0.6, (255, 255, 255), 1)

    return img


def show_detections(detections_, image_list, image_base, class_names, dest_path_, img_height, img_width):
    colors = ["#F28544", "#1DFA51", "#EDDC15", "#1E6AC2"]
    for i, d in enumerate(tqdm(detections_)):
        plt.figure(figsize=(20, 12))
        current_axis = plt.gca()
        plt.axis(False)
        image_ = imread(image_base + image_list[i])
        name = image_list[i]
        plt.imshow(image_)
        for box in d:
            xmin = box[2] * image_.shape[1] / img_width
            ymin = box[3] * image_.shape[0] / img_height
            xmax = box[4] * image_.shape[1] / img_width
            ymax = box[5] * image_.shape[0] / img_height
            class_ = int(box[0])
            conf = box[1]
            # print(class_)
            # print(conf)

            label = '{}: {:.2f}'.format(class_names[class_], conf)
            current_axis.add_patch(
                plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color=colors[class_], fill=False, linewidth=1))
            current_axis.text(xmin, ymin, label, size='x-small', color='white',
                              bbox={'facecolor': colors[class_], 'alpha': 1.0})

        plt.savefig(dest_path_ + name, dpi=100, bbox_inches="tight")


def show_detections_cv(detections_, image_list, image_base, class_names, dest_path_, img_height, img_width, line_width):
    colors = [(), (76, 153, 0), (35, 207, 233), (153, 76, 0)]
    gt_color = (255, 255, 255)
    show_gts = True

    for i, d in enumerate(tqdm(detections_)):
        # image_ = imread(image_base + image_list[i])
        image_ = cv2.imread(image_base + image_list[i])
        name = image_list[i]
        for box in d:
            xmin = int(box[2] * image_.shape[1] / img_width)
            ymin = int(box[3] * image_.shape[0] / img_height)
            xmax = int(box[4] * image_.shape[1] / img_width)
            ymax = int(box[5] * image_.shape[0] / img_height)
            class_ = int(box[0])
            cl = class_names[class_]
            conf = round(box[1], 2)
            color = colors[class_]

            # a = cv2.rectangle(image_, (xmin, ymin), (xmax, ymax), color, 1)
            # cv2.imshow("", a)

            cv2_draw_box_with_labels(image_, xmin, ymin, xmax, ymax, cl, conf, color, line_width)

        cv2.imwrite(dest_path_ + name, image_)


#     # Draw detection bboxes
#     for j in range(len(dets[i])):
#         d_xmin = int(dets[i][j][0])
#         d_ymin = int(dets[i][j][1])
#         d_xmax = int(dets[i][j][2])
#         d_ymax = int(dets[i][j][3])
#
#         cl = classes[(int(dets[i][j][4]))]
#         color = colors[int(dets[i][j][4])]
#         conf = round(dets[i][j][5], 2)
#
#         img = t.cv2_draw_box(img, d_xmin, d_ymin, d_xmax, d_ymax, color, line_width=1)
#
#         img = t.cv2_draw_box_with_labels(img_array=img,
#                                          xmin=d_xmin,
#                                          ymin=d_ymin,
#                                          xmax=d_xmax,
#                                          ymax=d_ymax,
#                                          class_=cl,
#                                          conf=conf,
#                                          bb_color=color,
#                                          line_width=1)
#
#

def process_predictions(detections):
    processed = []
    idx = [2, 3, 4, 5, 0, 1]
    for d in detections:
        if d.size == 0:
            processed.append(d)
        else:
            for i, x in enumerate(d):
                d[i] = d[i][idx]
            processed.append(d)
    return processed


def get_ground_truths(csv_path, image_column):
    gts = []
    df = pd.read_csv(csv_path)
    images = df[image_column].unique()
    for i in images:
        gt = df[df[image_column] == i]
        gt = gt[['xmin', 'ymin', 'xmax', 'ymax', 'class']]
        gt['difficult'] = 0
        gt['crowd'] = 0
        gt = gt.to_numpy()
        gts.append(gt)
    return gts


def get_mAP(preds_, gts_, n_classes, iou_thresh, show_coco=False):
    metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=n_classes)
    for i in range(len(preds_)):
        metric_fn.add(preds_[i], gts_[i])
    # compute PASCAL VOC metric
    print(
        f"VOC PASCAL mAP: {metric_fn.value(iou_thresholds=iou_thresh, recall_thresholds=np.arange(0., 1.1, 0.1))['mAP']}")

    # compute PASCAL VOC metric at the all points
    print(f"VOC PASCAL mAP in all points: {metric_fn.value(iou_thresholds=iou_thresh)['mAP']}")

    # compute metric COCO metric
    if show_coco:
        print(
            f"COCO mAP: {metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']}")


def pickle_data(data, file_path):
    file_ = open(file_path, 'ab')
    # source, destination
    pickle.dump(data, file_)
    file_.close()


def load_pickle(file_path):
    file_ = open(file_path, 'rb')
    data = pickle.load(file_)
    file_.close()

    return data


def check_duplicates(list_):
    if len(list_) == len(set(list_)):
        print('no duplicates')
    else:
        print("has duplicates")


def get_unique(list_):
    x = np.array(list_)
    unq = np.unique(x)
    print(unq)
    print(len(unq))
