# ALCE-FusionNet
### Data Prepare
* The dataset folder is stored in the following location: \<br>
Dataset/ \<br>
│ \<br>
├─—— image/   #RGB modality \<br>
│   ├─——train \<br>
│   └─——val  \<br>
│   └─——test  \<br>
├─—— images/   #IR modality  \<br>
│   ├─——train  \<br>
│   └─——val  \<br>
│   └─——test  \<br>
├─—— label/  \<br>
│   ├─——train  \<br>
│   └─——val  \<br>
│   └─——test  \<br>

### Train
'''
yolo task=detect    mode=train    model=ultralytics/cfg/datasets/yolov8-ALCE.yaml   data=ultralytics/cfg/datasets/m3fd.yaml  args...
'''

### Test
'''
yolo task=detect    mode=val    model=best.pt   data="Dateset/image/"   args...
'''
