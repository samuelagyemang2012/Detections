# Multimodal Object Detecton
This is a Keras port of the SSD model architecture introduced by Wei Liu et al. in the paper [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) 
which has been modified for multimodal object detection.

# Inputs
- RGB images
- RF images

## Input Data Examples
RGB and RF map pair used in training

|||
|---|---|
|![img01](./examples/c0000000071.jpg)|![img01](./examples/c000071_0000.npy.png)|

## Detection Results (Single modal vs Multi modal)
| | |  
|---|---|
| ![img01](./detection_results/s1.jpg)| ![img01](./detection_results/m1.jpg) |
| ![img01](./detection_results/s2.jpg)| ![img01](./detection_results/m2.jpg) |
| ![img01](./detection_results/s3.jpg)| ![img01](./detection_results/m3.jpg) |
| ![img01](./detection_results/s4.jpg)| ![img01](./detection_results/m4.jpg) |
| ![img01](./detection_results/s5.jpg)| ![img01](./detection_results/m5.jpg) |
| ![img01](./detection_results/s6.jpg)| ![img01](./detection_results/m6.jpg) |
| ![img01](./detection_results/s7.jpg)| ![img01](./detection_results/m7.jpg) |
| ![img01](./detection_results/s8.jpg)| ![img01](./detection_results/m8.jpg) |


## mAP
|**Model**|**VOC PASCAL mAP@0.5**|**COCO mAP@0.5**|
|-----|----------------------|----------------|
|Single modal model|0.087|0.045|
|Multi-modal model|0.169|0.120|

