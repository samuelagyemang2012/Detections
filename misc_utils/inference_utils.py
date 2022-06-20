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


def load_det_model(model_path_):
    ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)
    model = load_model(model_path_, custom_objects={'AnchorBoxes': AnchorBoxes,
                                                    'L2Normalization': L2Normalization,
                                                    'DecodeDetections': DecodeDetections,
                                                    'compute_loss': ssd_loss.compute_loss})
    return model


def do_detections(images_list_, image_base_path_, model_, img_height, img_width):
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
                                      confidence_thresh=0.5,
                                      iou_threshold=0.5,
                                      top_k=200,
                                      normalize_coords=True,
                                      img_height=img_height,
                                      img_width=img_width)

    return preds_decoded, orig_images


# def do_detections_multi(image_list_1_,image_list_2_,image_base_path_1_,image_base_path_2_,model_,img_height,img_width):
#     for i in range(0, len(image_list_1_)):
#         rgb_img = imre

def print_detections(detections_):
    for d in detections_:
        np.set_printoptions(precision=2, suppress=True, linewidth=90)
        print("Predicted boxes:\n")
        print('   class   conf xmin   ymin   xmax   ymax')
        print(d)


def cv_show_detections(detections_, image_list, image_base, class_names, dest_path_):
    for i, d in enumerate(detections_):
        image_ = cv2.imread(image_base + image_list[i])
        name = image_list[i]
        for box in d:
            xmin = box[2]
            ymin = box[3]
            xmax = box[4]
            ymax = box[5]

            image_ = cv2.rectangle(image_, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)


def cv_show_detection(detections_, image_, color, linewidth, class_names):
    for detection in detections_:
        for box in detection:
            print(box)
            xmin = int(box[2])
            ymin = int(box[3])
            xmax = int(box[4])
            ymax = int(box[5])

            image_ = cv2.rectangle(image_, (xmin, ymin), (xmax, ymax), color, linewidth)

    return image_


def show_detections(detections_, image_list, image_base, class_names, dest_path_, img_height, img_width):
    colors = ["#F28544", "#1DFA51", "#EDDC15", "#1E6AC2"]
    for i, d in enumerate(detections_):
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

            label = '{}: {:.2f}'.format(class_names[class_], conf)
            current_axis.add_patch(
                plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color=colors[class_], fill=False, linewidth=1))
            current_axis.text(xmin, ymin, label, size='x-small', color='white',
                              bbox={'facecolor': colors[class_], 'alpha': 1.0})

        plt.savefig(dest_path_ + name, dpi=100, bbox_inches="tight")
