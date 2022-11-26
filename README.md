# effective-octo-guacamole
AI ML project - Assisted Navigation for the Visually Impaired using Object recognition

## YOLO (You Only Look Once)
 
Download the pre-trained YOLO v3 weights file from this [link](https://pjreddie.com/media/files/yolov3.weights) and place it in the current directory or you can directly download to the current directory in terminal using
 
 `$ wget https://pjreddie.com/media/files/yolov3.weights`

Yolo code terminal command (old)
`$ python3 yolo_image.py --image Images/<file_name>.jpg --config Yolo/yolov3.cfg --weights Yolo/yolov3.weights --classes Yolo/yolov3.txt`

Yolo code terminal command (new)
`$ python3 yolo_image.py --image Images/<file_name>.jpg`

Haar Cascade code terminal command
`$ python3 hc_image.py --image Images/<file_name>.jpg`
