from ultralytics import YOLO

# Params:
### train (bool) - to train the model or not 
### sole_od (bool) - if train = True, then train a sole bounding box detection network
###                  if train = False, then use a sole bounding box detection network 
### od_color (bool) - if train = True, then train a dual bounding box and color detection network
###                   if train = False, then use a dual bounding box and color detection netork 
train = False
sole_od = False
od_color = True 

# Use base model for pre-trained network, else, currently, use bounding box detection network
if train: 
    model_path = "D:/Betafinder/Betafinder/models/object_detection/yolo_weights/base.pt"
else:
    if sole_od:
        model_path = "D:/Betafinder/Betafinder/models/object_detection/yolo_weights/bounding_box.pt"
    elif od_color: 
        model_path = "D:/Betafinder/Betafinder/models/object_detection/yolo_weights/bounding_and_color.pt"
    else:
        raise ValueError("Model must be either sole_bounding_box or bounding_and_color.")

# Dataset paths for training bounding box detection
dataset_od_paths = [
    "D:/Betafinder/data/train_sole_bounding/New Hold Detections.v10-june-8-final-dataset.yolov8",
    "D:/Betafinder/data/train_sole_bounding/just hold.v3i.yolov8",
    "D:/Betafinder/data/train_sole_bounding/Hold Detector.v13i.yolov8",
    "D:/Betafinder/data/train_sole_bounding/Hold Detector.v4i.yolov8"
]

# Dataset path for training bounding box and color detection
dataset_od_color_paths = [
    "D:/Betafinder/data/train_with_color/hold.v2i.yolov8/data.yaml",
    "D:/Betafinder/data/train_with_color/Annotated Hold Detection.v1i.yolov8/data.yaml",
    "D:/Betafinder/data/train_with_color/Annotated Hold Detection.v2-color-holds.yolov8/data.yaml",
    "D:/Betafinder/data/train_with_color/anooted/data.yaml",
    "D:/Betafinder/data/train_with_color/Holds recognizer.v1i.yolov8/data.yaml",
    "D:/Betafinder/data/train_with_color/New Hold Detections.v10-june-8-final-dataset.yolov8/data.yaml"
]

# initialize model
model = YOLO(model_path)

# Train if necessary
if train and sole_od:
    for train_path in dataset_od_paths:
        model.train(data = train_path, epochs = 10)
elif train and od_color:
    for train_path in dataset_od_color_paths:
        model.train(data = train_path, epochs = 10)
else:
    raise ValueError("When training, model must be either sole_bounding_box or bounding_and_color.")

# metrics = model.val()
results = model("D:/Betafinder/data/Object_detection_holds/IMG20240218163716.jpg", save = True, save_txt = True)

print(results)