# EventKitchen_toolkit
This is the toolkit of the EventKitchen dataset from our ECCV 2026 paper ***"Cooking beyond Frames: A Stereo Event Camera Dataset in the Kitchen."*** In this toolkit, we present how to load the data and prepare data for action recognition, object detection, and stereo depth estimation.

<p align="center">
  <a href="https://arxiv.org/abs/2608.04865"><img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=white" alt="arXiv"></a>&nbsp;&nbsp;&nbsp;
  <a href="https://chengmingf.github.io/EventKitchen.github.io/"><img src="https://img.shields.io/badge/Project-Page-1f6f68?logo=githubpages&logoColor=white" alt="Project Page"></a>&nbsp;&nbsp;&nbsp;
  <a href="https://data.4tu.nl/datasets/583f8a2f-5448-4a9f-84c7-caaca92b1835"><img src="https://img.shields.io/badge/4TU-Dataset-0076a8" alt="Dataset"></a>
</p>

## Table of Contents

- [Load Events](#load-events)
- [Action Recognition](#action-recognition)
- [Object Detection](#object-detection)
- [Stereo Depth Estimation](#stereo-depth-estimation)
- [Cite us](#cite-us)

## Load data
Please follow the [dataset download page](https://chengmingf.github.io/EventKitchen.github.io/download.html) to download the dataset and refer to the dataset structure. The collected events are saved in the [DSEC format](https://dsec.ifi.uzh.ch/data-format/). Please refer to the [read_event_file.ipynb](read_event_file.ipynb) for loading events.

### Dataset split
We split the dataset in session level, the training set including Session 1, 2, 5, 6, 7, 8, 9, 11, 12, 13; the test set including Session 3, 4, 10, 14. Also saved in [dataset_split.csv](dataset_split.csv)

## Action Recognition
<!-- We introduce the groud truth and baselines for action recognition here. -->
### Action Segment
The action annotations are saved in the .csv files. The format is shown below:
| Full_action_label | Action_label | Verb | Object | Global_start_time | Global_end_time | Length   |
| --- | --- | --- | --- | --- | --- | --- |
|Take fruit         | Take fruit | Take | fruit | 1704552331.171166 | 1704552336.109683 | 4.938517 |
|Take bowl          | Take bowl	| Take | bowl | 1704552356.661439 | 1704552359.6558 | 2.994361 |
|Put fruit into bowl | Put fruit | Put | fruit | 1704552368.470344 | 1704552370.737917 | 2.267573 |
|Throw waste into bin | Throw waste | Throw | waste | 1704552575.00379 | 1704552583.475926 | 8.472136 |
| ... | ... | ... | ... | ... | ... | ... |

- **Full_action_label**: the full label of the action.
- **Action_label**: composed as "Verb" + "Noun". This label used to train the **Action Recognition** baselines.
- **Verb**: the verb component in the action label.
- **Object**: the noun component in the action label.
- **Global_start_time**: the global timestamp to locate the start of the action in multi-modal recordings, in the unit of second.
- **Global_end_time**: the end timestamp to locate the start of the action in multi-modal recordings, in the unit of second.
- **Length**: the time length of the action segment.

### Prepare data
To prepare the data, we use the global start and end timestamp to align the events and action segments. And as reported in the paper, we use a subset of 69 actoin classes to train baselines, as shown in [action_subset.csv](action_subset.csv). Please refer to the [prepare_action_recognition.ipynb](prepare_action_recognition.ipynb) for loading action segments.

### Baselines
We evaluate [TSM](https://github.com/mit-han-lab/temporal-shift-module) and [Swin](https://github.com/SwinTransformer/Video-Swin-Transformer) for action recognition as reported in the paper. To reproduce the results, please refer to their official github repo. And all implementation details are reported in the supplemetary matirals.

## Object Detection
<!-- We introduce the groud truth and baselines for object detection here. -->
### Bounding Box
Bounding boxes are saved in .csv files. The format is shown below:
| ts | bbox |
| --- | --- |
| 1704552350.708145 | [{'x': 848, 'y': 127, 'w': 337, 'h': 326, 'class': 'Box'}] |
| 1704552366.798267 | [{'x': 523, 'y': 15, 'w': 577, 'h': 580, 'class': 'Bowl'}] |
| 1704552370.80641 | [{'x': 376, 'y': 186, 'w': 288, 'h': 338, 'class': 'Box'}, {'x': 429, 'y': 8, 'w': 499, 'h': 573, 'class': 'Bowl'}] |
| 1704552374.8697 | [{'x': 558, 'y': 14, 'w': 585, 'h': 539, 'class': 'Bowl'}, {'x': 453, 'y': 436, 'w': 331, 'h': 276, 'class': 'Box'}] |
| ... | ... |

- **ts**: the global timestamp of the bounding box, in the unit of second.
- **bbox**: the bounding boxes in the dictionary, with  
    {'x': x-axis of the topleft corner,  
     'y': y-axis of the topleft corner,  
     'w': width of the bounding box,  
     'h': height of the bounding box,  
     'class': object class of the bounding box}.  

### Prepare data
To prepare the data, we use the global timestamp to align the events and bounding boxes. Please refer to the [prepare_object_detection.ipynb](prepare_object_detection.ipynb) for loading bounding boxes.

### Baselines
We evaluate [YOLOv10](https://github.com/THU-MIG/yolov10), [RVT](https://github.com/uzh-rpg/rvt), and [EvRT-DETR](https://github.com/realtime-intelligence/evrt-detr) for object detection as reported in the paper. To reproduce the results, please refer to their official github repo. And all implementation details are reported in the supplemetary matirals.

## Stereo Depth Estimation

### Extract Depth Maps
The depth maps are saved in .avi videos. And the corrsponding global timestamp per depth map is saved in .csv file as:
| frame_id | timestamp |
| --- | --- |
| 0 | 1717341546.164076 |
| 1 | 1717341546.231737 |
| 2 | 1717341546.294798 |
| 3 | 1717341546.355427 |
| ... | ... |

Please refer to the [extract_depth_map.ipynb](extract_depth_map.ipynb) to extract depth maps.

### Rectification
To conduct the stereo depth estimation, you need to firstly project the depth map to the left event camera, then recitify the left event camera, right event camera camera, and
the projected depth. Follow the listed steps to prepare the data:

1. Download the calibration results and unzip, see [Project Page](https://chengmingf.github.io/EventKitchen.github.io/download.html).
2. We provide a [python script](calibration_loader/EventKitchen_Calibration.py) for loading the calibration matrix. To rectify the depth and events, please refer to the [preapare_stereo_depth_estimation.ipynb](preapare_stereo_depth_estimation.ipynb).

### Baselines
We evaluate [SE-CFF](https://github.com/yonseivnl/se-cff) and [FoundationStereo](https://github.com/NVlabs/FoundationStereo) for stereo depth estimation as reported in the paper. To reproduce the results, please refer to their official github repo. And all implementation details are reported in the supplemetary matirals. And as reported in the paper, we use a subset of the recorded sessions to evaluate the stereo depth estimation. The subset is shown as:

**Training set** [train_depth_map.csv](train_depth_map.csv)

| Session 01 | Session 02 | Session 05 | Session 06 | Session 07 | Session 08 | Session 09 | Session 11 | Session 12 | Session 13 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| vege_salad | cereal_bowl | fry_bacon | sandwich | fry_egg | lemon_water | sandwich | cut_bread | fry_pepper | fruit_salad |
| cereal_bowl | cut_bread | fry_egg | tea_1 | cut_cake | coffee | wash_dish | sandwich | fry_bacon | vege_salad |
| - | - | - | cut_bread | - | - | tea | - | - | - |

**Test set** [test_depth_map.csv](test_depth_map.csv)

| Session 03 | Session 04 | Session 10 | Session 14 |
| :---: | :---: | :---: | :---: |
| cut_bread | coffee | fry_pepper | wash_dish |
| tea_2 | vege_salad | fry_egg | cereal_bowl |
| coffee | - | - | cut_bread |

## Cite us
If you use EventKitchen in your research, please cite:
```bibtex
@article{feng2026cooking,
  title={Cooking beyond Frames: A Stereo Event Camera Dataset in the Kitchen},
  author={Feng, Chengming and Araghi, Hesam and Zheng, Liming and Dupeyroux, Julien and Zhang, Xucong and van Gemert, Jan and T{\"o}men, Nergis},
  journal={arXiv preprint arXiv:2608.04865},
  year={2026}
}
```
