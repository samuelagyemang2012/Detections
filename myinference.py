import os
from tensorflow.keras import backend as K
import misc_utils.inference_utils as msu

img_height = 300
img_width = 480
confidence_threshold = 0.4
classes = ['background', 'car', 'truck', 'pedestrian', 'bicyclist', 'light']

model_path = "C:/Users/Sam/Desktop/xxx.h5"
base_path = "D:/Datasets/subsets/"  # "C:/Users/Administrator/Desktop/Self Driving Cars/images/"
dest_path = "C:/Users/Sam/Desktop/detections/ssd/"

images_list = os.listdir(base_path)
K.clear_session()

model = msu.load_det_model(model_path)
detections, orig_sizes = msu.do_detections(images_list, base_path, model, img_height, img_width)
msu.print_detections(detections)
msu.show_detections(detections, images_list, base_path, classes, dest_path, img_height, img_width)
