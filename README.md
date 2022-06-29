## SSD: Single-Shot MultiBox Detector implementation in Keras
---
### Contents
### Source: https://github.com/pierluigiferrari/ssd_keras


### Overview

This is a Keras port of the SSD model architecture introduced by Wei Liu et al. in the paper [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325).

### Detections

Below are some prediction of the trained original SSD300 model.
This model was trained on the [Udacity Self Driving cars dataset](https://www.kaggle.com/datasets/sshikamaru/udacity-self-driving-car-dataset).
The detection results are seen below.

| | |
|---|---|
| ![img01](detection_results/1478020998721490820.jpg)| ![img01](detection_results/1478021584075204499.jpg) |
| ![img01](detection_results/a6.jpg) | ![img01](detection_results/a3.jpg) |
| ![img01](detection_results/a1.jpg) | ![img01](detection_results/Capture.PNG) |
| ![img01](detection_results/a7.jpeg) | ![img01](detection_results/dublin.png) |
| ![img01](detection_results/6.png) | ![img01](detection_results/48.png) |

# Multimodal Object Detecton
The SSD model architecture has been modified for multimodal object detection using RGB images and range-azimuth frequency maps. 
The multimodal SSD object detector is trained on the [CRUW dataset](https://www.cruwdataset.org/introduction) and consists of 3 classes (cars, cyclist & pedestrian). 

## Input Data Examples
An example of the RGB image and range-azimuth map pairs used for training.

|||
|---|---|
|![img01](./data_examples/c0000000071.jpg)|![img01](./data_examples/c000071_0000.npy.png)|

## Detection Results-Single modal (RGB images only) vs Multimodal input
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
|**Model**|mAP
|-----|----------------------|
|Single modal model|44.83%
|Multi-modal model|**80.71%**|

|||
|-----|-----|
|![img01](./detection_results/mAP_single.png)|![img01](./detection_results/mAP_multi.png)|



### Dependencies

* Python 3.x
* Numpy
* TensorFlow 1.x
* Keras 2.x
* OpenCV
* Beautiful Soup 4.x
