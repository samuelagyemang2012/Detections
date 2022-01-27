import pandas as pd

train_df = pd.read_csv("C:/Users/Administrator/Desktop/Self Driving Cars/labels_train.csv")
val_df = pd.read_csv("C:/Users/Administrator/Desktop/Self Driving Cars/labels_val.csv")


def shuffle_df(df):
    return df.sample(frac=1).reset_index(drop=True)


# print(train_df)
# print(val_df)

# 3, pedestrian
# 5, light

tr_ped_light_df = train_df[(train_df["class_id"] == 3) | (train_df["class_id"] == 5)]
vl_ped_light_df = val_df[(val_df["class_id"] == 3) | (val_df["class_id"] == 5)]

tr_ped_light_df = shuffle_df(tr_ped_light_df)
vl_ped_light_df = shuffle_df(vl_ped_light_df)

print(tr_ped_light_df)
print(vl_ped_light_df)

# subset_train = train_df[0:30000]
# subset_val = val_df[0:4000]
#
# print(len(subset_train['frame'].unique()))
# print(len(subset_val['frame'].unique()))
#
# print(subset_train['class_id'].unique())
# print(subset_val['class_id'].unique())
#
tr_ped_light_df.to_csv("C:/Users/Administrator/Desktop/Self Driving Cars/lights_peds/train_subset.csv", index=None)
vl_ped_light_df.to_csv("C:/Users/Administrator/Desktop/Self Driving Cars/lights_peds/val_subset.csv", index=None)
