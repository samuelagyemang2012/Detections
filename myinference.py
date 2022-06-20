import os
from tensorflow.keras import backend as K
import misc_utils.inference_utils as msu
import pandas as pd

img_height = 448
img_width = 448
confidence_threshold = 0.4
classes = ['background', 'car', 'bicyclist', 'pedestrian']

model_path = "C:/Users/Administrator/Desktop/cruw_model.h5"
base_path = "C:/Users/Administrator/Desktop/datasets/CRUW/ALL/resized/images/"
data_path = "data/cruw_test_rgb.csv"
dest_path = "C:/Users/Administrator/Desktop/my_detections/cruw/single/"

df = pd.read_csv(data_path)
images_list = df['image'].to_list()
images_list = images_list[0:40]

# images_list = os.listdir(base_path)
K.clear_session()

model = msu.load_det_model(model_path)
detections, orig_sizes = msu.do_detections(images_list, base_path, model, img_height, img_width)
msu.print_detections(detections)
msu.show_detections(detections, images_list, base_path, classes, dest_path, img_height, img_width)
print("Done")
