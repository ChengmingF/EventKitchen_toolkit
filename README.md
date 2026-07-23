# EventKitchen_toolkit
This is the toolkit of the EventKitchen dataset from "Cooking beyond Frames: A Stereo Event Camera Dataset in the Kitchen." 

## Dataset Structure
The dataset is collected with a stereo pair of Prophesee GEN4 cameras, a stereo pair of CMOS RGB cameras, an Intel RealSense D435i/D455, and a 9-axis IMU. Please refer to the [Project Page](https://chengmingf.github.io/EventKitchen.github.io/) to check the dataset structure.

## Load Events
The collected events are saved in the [DSEC format](https://dsec.ifi.uzh.ch/data-format/), please refer to this [python script](event_reader/eventslicer.py) to load events. You can easily load events as below:
```python
import h5py
import hdf5plugin
from event_reader.eventslicer import EventSlicer

# load events
event_file = "" # path to the dataset/LeftEvent/LeftEvent.hdf5 or dataset/RightEvent/RightEvent.hdf5
event_loader = EventSlicer(h5py.File(event_file, 'r'))

start_time = # the global start timestamp of a desired event slice, unit is microsecond
end_time = # the global end timestamp of a desired event slice, unit is microsecond
event_slice = event_loader.get_events(start_time, end_time)
```

## Action Recognition
We introduce the groud truth and baselines for action recognition here.
### Action Segment
The action annotations are saved in the .csv format as below:
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

To prepare the data, use the global start and end timestamp to align the events and action segments. We provide an example to get the aligned data below:
```python
import h5py
import hdf5plugin
import pandas as pd
import numpy as np
from event_reader.eventslicer import EventSlicer

# load action labels
action_label_file = "" # path to the Annotations/actions/cooking_activity_action.csv
action_labels = pd.read_csv(action_label_file)

# load events
event_file = "" # path to the dataset/LeftEvent/LeftEvent.hdf5 or dataset/RightEvent/RightEvent.hdf5
event_loader = EventSlicer(h5py.File(event_file, 'r'))
   
# align events and action
for index, label in action_labels["Action_label"].items():
    start_time = action_labels.loc[index, "Global_start_time"] * 1e6 # convert seconds to microseconds
    end_time = action_labels.loc[index, "Global_end_time"] * 1e6 # convert seconds to microseconds
    event_slice = event_loader.get_events(start_time, end_time)
    aligned_data = [label, event_slice]

    #####
    ## your own data processing
    #####

```
### Baselines
We evaluate [TSM](https://github.com/mit-han-lab/temporal-shift-module) and [Swin](https://github.com/SwinTransformer/Video-Swin-Transformer) for action recognition as reported in the paper. To reproduce the results, please refer to their official github repo. And all implementation details are reported in the supplemetary matirals.

## Object Detection
We introduce the groud truth and baselines for object detection here.
### Bounding Box
Bounding boxes are saved in .csv files as below:
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

To prepare the data, use the global timestamp to align the events and bounding boxes. We provide an example to get the aligned data below:
```python
import h5py
import hdf5plugin
import pandas as pd
import numpy as np
from event_reader.eventslicer import EventSlicer

# load action labels
bbox_label_file = "" # path to the Annotations/objects/cooking_activity_LeftEvent_bbox.csv or Annotations/objects/cooking_activity_RightEvent_bbox.csv
bbox_labels = pd.read_csv(bbox_label_file)

# load events
event_file = "" # path to the dataset/LeftEvent/LeftEvent.hdf5 or dataset/RightEvent/RightEvent.hdf5
event_loader = EventSlicer(h5py.File(event_file, 'r'))
   
# align events and action
for index, ts in bbox_labels["ts"].items():
    delta_T = 0.033 # the event slice length to align the bounding boxes, unit is second
    start_time = (ts - delta_T) * 1e6 # convert seconds to microseconds
    end_time = ts * 1e6 # convert seconds to microseconds
    label = bbox_labels.loc[index, "bbox"]
    event_slice = event_loader.get_events(start_time, end_time)
    aligned_data = [label, event_slice]

    #####
    ## your own data processing
    #####
```

### Baselines
We evaluate [YOLOv10](https://github.com/THU-MIG/yolov10), [RVT](https://github.com/uzh-rpg/rvt), and [EvRT-DETR](https://github.com/realtime-intelligence/evrt-detr) for object detection as reported in the paper. To reproduce the results, please refer to their official github repo. And all implementation details are reported in the supplemetary matirals.

## Stereo Depth Estimation
### Depth Maps
The depth maps are saved in .avi videos. The corrsponding global timestamp per depth map is saved in .csv file as:
| frame_id | timestamp |
| --- | --- |
| 0 | 1717341546.164076 |
| 1 | 1717341546.231737 |
| 2 | 1717341546.294798 |
| 3 | 1717341546.355427 |
| ... | ... |

We provide an example to extract depth maps from the video as below:
```python
import cv2
import pandas as pd
from pathlib import Path

# load depth video
depth_video = "" # path to the dataset/DEPTH/DEPTH.avi
depth_timestamp_file = "" # path to the dataset/DEPTH/DEPTH_timestamp.avi
depth_timestamp = pd.read_csv(depth_timestamp_file)

# save path
output_dir = Path("") # path to save the depth frames
output_dir.mkdir(parents=True, exist_ok=True)

# start converting, the saved depth maps are in uint16
cap = cv2.VideoCapture(
    depth_video,
    cv2.CAP_FFMPEG,
    [cv2.CAP_PROP_CONVERT_RGB, 0],
)

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {depth_video}")
frame_idx = 0

while True:
    success, frame = cap.read()

    if not success:
        break

    if frame.dtype != "uint16":
        raise TypeError(
            f"Expected uint16 depth, but OpenCV returned {frame.dtype} "
            f"with shape {frame.shape}"
        )

    frame_ts = depth_timestamp.loc[frame_idx, "timestamp"]
    output_path = output_dir / f"{frame_ts:.06f}.tiff"

    if not cv2.imwrite(str(output_path), frame):
        raise RuntimeError(f"Failed to save: {output_path}")

    frame_idx += 1

cap.release()
```
### Calibration
We provide the calibration of 