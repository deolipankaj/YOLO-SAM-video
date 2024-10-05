# YOLO-SAM-video
The following code processes a video through YOLOv8 and SAM and saves various information such as binary masks, segmentation outputs, etc.,
The main intention behind this is not to inference (for real-time purposes), but rather testing your pre-trained model on any video and saving the data for e.g. further or fine tuning.


# Installation

1. Clone the repository
   ```bash
   git clone https://github.com/deolipankaj/YOLO-SAM-video.git
   ```
 2. Create conda environment
    ```bash
    conda env create -f environment.yml
    ```
  3. Run
     ```bash
     python yolo-sam-video-multiple-classes.py
     ```
     
# Options
The code will save the following information i.e.

1. Original frames from the video
2.  Corresponding masked segmented output
3. Corresponding binary masks segmentation
4. Corresponding bounding box labels in YOLO format

# Pre-requisites
1. Pretrained model in yolo-format (e.g. ```yolov8m.pt```) (https://docs.ultralytics.com/models/yolov8/#key-features)
2. Pretraianed SAM model (e.g. ```sam_vit_h_4b8939.pth```) (https://github.com/facebookresearch/segment-anything)
3. Input video

You have 3 different options i.e.

## Custom trained model (single class)
If you have trained a custom model (with one class e.g. tree here), you can use ```yolo-sam-video-single-class.py``` with your pretrained yolo model and the existing SAM model.

![example_original_images](https://github.com/user-attachments/assets/88e5873f-dcfb-4338-885c-438582428bdc)
![example_segmented_output](https://github.com/user-attachments/assets/68f8ab99-f7a4-4ad7-98ae-47afc44b5191) 
![example_binary_masks](https://github.com/user-attachments/assets/447c1740-2582-48ca-8a90-ce574bf64e49)


## Custom trained model (multiple classes)
For multiple classes, simply replace the class mappings with your classes along-with your pretrained model.

## Pretrained YOLOV8 on COCO and SAM
Currently, the code will take a video as an input, process it through YOLOV8 (Pretrained on COCO) and SAM and save multiple things .

![example_original_images_compressed](https://github.com/user-attachments/assets/92df9f46-a849-4525-b857-af7e0d4545f1)
![example_segmented_output](https://github.com/user-attachments/assets/a84c7ee2-5335-461b-b84c-1bde3688688b) 
![example_binary_masks](https://github.com/user-attachments/assets/0ccced73-4a77-4e49-be94-cded3b612a8d)
