from tensorflow.keras import backend as K
import misc_utils.inference_utils as msu
import pandas as pd

img_height = 448
img_width = 448
confidence_threshold = 0.5
iou_threshold = 0.5
classes = ['background', 'car', 'bicyclist', 'pedestrian']
model_path = "C:/Users/Administrator/Desktop/cruw_model_v2.h5"
image_base_path = "C:/Users/Administrator/Desktop/datasets/CRUW/ALL/resized/images/"
data_path = "data/cruw_test_data_rgb.csv"
dest_path = "C:/Users/Administrator/Desktop/my_detections/cruw/single/detections_cv/"

# select data
df = pd.read_csv(data_path)

# images_list = df['image'].to_list()
images_list = df['image'].unique()
print(len(images_list))

# do detections
K.clear_session()
model = msu.load_det_model(model_path)
print("model loaded")

detections = msu.do_detections(images_list,
                               image_base_path,
                               model,
                               img_height,
                               img_width,
                               confidence_threshold,
                               iou_threshold)
print(len(detections))
# msu.print_detections(detections)

# save detections
# msu.show_detections(detections, images_list, image_base_path, classes, dest_path, img_height, img_width)
msu.show_detections_cv(detections, images_list, image_base_path, classes, dest_path, img_height, img_width, 1)

print("Done")
