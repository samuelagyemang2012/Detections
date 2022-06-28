from tensorflow.keras import backend as K
import pandas as pd
import misc_utils.inference_utils as msu

img_height = 448
img_width = 448
confidence_threshold = 0.5
iou_threshold = 0.5
classes = ['background', 'car', 'bicyclist', 'pedestrian']
multi_model_path = "C:/Users/Administrator/Desktop/cruw_model_multi_v2.h5"
rgb_img_base_path = "C:/Users/Administrator/Desktop/datasets/CRUW/ALL/resized/CRUW_kaggle/images/"
rf_img_base_path = "C:/Users/Administrator/Desktop/datasets/CRUW/ALL/resized/CRUW_kaggle/rf_maps/"
rgb_data_path = "data/cruw_test_data_rgb.csv"
rf_data_path = "data/cruw_test_data_rf.csv"
# dest_path = "C:/Users/Administrator/Desktop/my_detections/cruw/multi/lab/"

# select data
rgb_df = pd.read_csv(rgb_data_path)
rf_df = pd.read_csv(rf_data_path)

rgb_list = rgb_df['image'].unique()
rf_list = rf_df['image'].unique()

# load model
K.clear_session()
multi_model = msu.load_det_model(multi_model_path)
detections = msu.do_multi_detections(rgb_list,
                                     rgb_img_base_path,
                                     rf_list,
                                     rf_img_base_path,
                                     multi_model,
                                     img_height,
                                     img_width,
                                     confidence_threshold,
                                     iou_threshold)

gts = msu.get_ground_truths(rgb_data_path, 'image')
processed_detections = msu.process_predictions(detections)

print(len(gts))
print(len(processed_detections))

# msu.pickle_data(gts, "C:/Users/Administrator/Desktop/gts_" + str(limit) + ".sav")
msu.pickle_data(processed_detections, "C:/Users/Administrator/Desktop/multi_preds.sav")
print("Done")
