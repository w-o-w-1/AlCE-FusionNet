# ALCE-FusionNet
### Data Prepare
* The dataset folder is stored in the following location:
Dataset/
│
├─—— image/   #RGB modality
│   ├─——train
│   └─——val
│   └─——test
├─—— images/   #IR modality
│   ├─——train
│   └─——val
│   └─——test
├─—— label/
│   ├─——train
│   └─——val
│   └─——test

### Train
'''
yolo task=detect    mode=train    model=ultralytics/cfg/datasets/yolov8-ALCE.yaml   data=ultralytics/cfg/datasets/m3fd.yaml  args...
'''

### Test
'''
yolo task=detect    mode=val    model=best.pt   data="Dateset/image/"   args...
'''
