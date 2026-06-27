# EventKitchen_toolkit
This is the toolkit of the EventKitchen dataset from "Cooking beyond Frames: A Stereo Event Camera Dataset in the Kitchen." 

## Load Events
The collected events are saved in the [DSEC format](https://dsec.ifi.uzh.ch/data-format/), please refer to this [python script](event_reader/eventslicer.py) to load events. 

## Load Annotations
### Action Segments
The action segments are saved in the .csv format as below:
| Full_action_label | Action_label | Verb | Object | Global_start_time | Global_end_time   | length   |
| ---               | ---          | ---  | ---    | ---               | ---               | ---      |
|Take fruit	        |Take fruit	   |Take  |	fruit  | 1704552331.171166 | 1704552336.109683 | 4.938517 |

## Load Calibration

