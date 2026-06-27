# EventKitchen_toolkit
This is the toolkit of the EventKitchen dataset from "Cooking beyond Frames: A Stereo Event Camera Dataset in the Kitchen." 

## Load Events
The collected events are saved in the [DSEC format](https://dsec.ifi.uzh.ch/data-format/), please refer to this [python script](event_reader/eventslicer.py) to load events. 

## Load Annotations / Ground Truth
### Action Segment
The action annotations are saved in the format as below:
| Full_action_label | Action_label | Verb | Object | Global_start_time | Global_end_time | length   |
| --- | --- | --- | --- | --- | --- | --- |
|Take fruit         | Take fruit | Take | fruit | 1704552331.171166 | 1704552336.109683 | 4.938517 |
|Take bowl          | Take bowl	| Take | bowl | 1704552356.661439 | 1704552359.6558 | 2.994361 |
|Put fruit into bowl | Put fruit | Put | fruit | 1704552368.470344 | 1704552370.737917 | 2.267573 |
|Throw waste into bin | Throw waste | Throw | waste | 1704552575.00379 | 1704552583.475926 | 8.472136 |
| ... | ... | ... | ... | ... | ... | ... |

### Bounding Box
Bounding boxes are saved in csv files as follows:
| ts | bbox |
| --- | --- |
| 1704552350.708145 | [{'x': 848, 'y': 127, 'w': 337, 'h': 326, 'class': 'Box'}] |
| 1704552366.798267 | [{'x': 523, 'y': 15, 'w': 577, 'h': 580, 'class': 'Bowl'}] |
| 1704552370.80641 | [{'x': 376, 'y': 186, 'w': 288, 'h': 338, 'class': 'Box'}, {'x': 429, 'y': 8, 'w': 499, 'h': 573, 'class': 'Bowl'}] |
| 1704552374.8697 | [{'x': 558, 'y': 14, 'w': 585, 'h': 539, 'class': 'Bowl'}, {'x': 453, 'y': 436, 'w': 331, 'h': 276, 'class': 'Box'}] |
| ... | ... |


