import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
import os

# COCO class names
coco_class_mapping = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    4: 'airplane',
    5: 'bus',
    6: 'train',
    7: 'truck',
    8: 'boat',
    9: 'traffic light',
    10: 'fire hydrant',
    11: 'stop sign',
    12: 'parking meter',
    13: 'bench',
    14: 'bird',
    15: 'cat',
    16: 'dog',
    17: 'horse',
    18: 'sheep',
    19: 'cow',
    20: 'elephant',
    21: 'bear',
    22: 'zebra',
    23: 'giraffe',
    24: 'backpack',
    25: 'umbrella',
    26: 'handbag',
    27: 'tie',
    28: 'suitcase',
    29: 'frisbee',
    30: 'skis',
    31: 'snowboard',
    32: 'sports ball',
    33: 'kite',
    34: 'baseball bat',
    35: 'baseball glove',
    36: 'skateboard',
    37: 'surfboard',
    38: 'tennis racket',
    39: 'bottle',
    40: 'wine glass',
    41: 'cup',
    42: 'fork',
    43: 'knife',
    44: 'spoon',
    45: 'bowl',
    46: 'banana',
    47: 'apple',
    48: 'sandwich',
    49: 'orange',
    50: 'broccoli',
    51: 'carrot',
    52: 'hot dog',
    53: 'pizza',
    54: 'donut',
    55: 'cake',
    56: 'chair',
    57: 'couch',
    58: 'potted plant',
    59: 'bed',
    60: 'dining table',
    61: 'toilet',
    62: 'tv',
    63: 'laptop',
    64: 'mouse',
    65: 'remote',
    66: 'keyboard',
    67: 'cell phone',
    68: 'microwave',
    69: 'oven',
    70: 'toaster',
    71: 'sink',
    72: 'refrigerator',
    73: 'book',
    74: 'clock',
    75: 'vase',
    76: 'scissors',
    77: 'teddy bear',
    78: 'hair drier',
    79: 'toothbrush',
}

def save_polygons_txt(masks, frame_name, polygons_dir, image_width, image_height, class_ids):
    """
    Save mask contours as polygons in text format with normalized coordinates.
    """
    polygon_file_path = os.path.join(polygons_dir, f'{frame_name}_polygons.txt')

    with open(polygon_file_path, 'w') as file:
        for i, (mask, class_id) in enumerate(zip(masks, class_ids)):
            mask_np = mask.cpu().numpy().astype(np.uint8)  # Convert mask to binary image
            _, mask_np = cv2.threshold(mask_np, 0.5, 1, cv2.THRESH_BINARY)

            if np.sum(mask_np) == 0:
                continue  # Skip empty masks

            contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                normalized_polygon = []
                for point in contour.reshape(-1, 2):
                    normalized_x = point[0] / image_width
                    normalized_y = point[1] / image_height
                    normalized_polygon.append((normalized_x, normalized_y))
                # Write each polygon as a new line in the text file with class ID
                class_name = coco_class_mapping.get(int(class_id), 'Unknown')
                polygon_str = f"{class_name} " + ' '.join([f'{x} {y}' for x, y in normalized_polygon])
                file.write(polygon_str + '\n')
    
    print(f"Saved polygons in text format: {polygon_file_path}")

def save_binary_masks(masks, frame_name, mask_dir, class_ids):
    """
    Save each individual binary mask as a separate PNG file, labeled with the class name.
    """
    for i, (mask, class_id) in enumerate(zip(masks, class_ids)):
        mask_np = mask.cpu().numpy().astype(np.uint8) * 255  # Convert mask to binary image
        class_name = coco_class_mapping.get(int(class_id), 'Unknown')
        mask_path = os.path.join(mask_dir, f'{frame_name}_{class_name}_mask_{i}.png')
        cv2.imwrite(mask_path, mask_np)
        print(f"Saved binary mask: {mask_path}")

def save_labels(predicted_boxes, predicted_classes, frame_name, labels_dir, image_width, image_height):
    """
    Save bounding boxes in a text file with normalized coordinates and class IDs.
    """
    label_path = os.path.join(labels_dir, f'{frame_name}.txt')
    
    with open(label_path, 'w') as f:
        for box, class_id in zip(predicted_boxes, predicted_classes):
            x0, y0, x1, y1 = box
            x_center = ((x0 + x1) / 2) / image_width
            y_center = ((y0 + y1) / 2) / image_height
            width = (x1 - x0) / image_width
            height = (y1 - y0) / image_height

            # Write class_id and bounding box coordinates
            f.write(f"{int(class_id)} {x_center} {y_center} {width} {height}\n")

def process_frame(frame, model, predictor, device, frame_name, output_dir, mask_dir, labels_dir, polygons_dir):
    try:
        # Convert frame to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_height, image_width = image.shape[:2]

        # YOLO object detection
        with torch.no_grad():
            results = model.predict(source=image, conf=0.20, device=device)
        
        # Check if any objects were detected
        if len(results) == 0 or len(results[0].boxes) == 0:
            print(f"No objects detected in frame: {frame_name}")
            return

        # Get bounding boxes and class IDs
        boxes = results[0].boxes.xyxy  # Bounding boxes in (x0, y0, x1, y1) format
        class_ids = results[0].boxes.cls  # Class IDs

        # Convert to lists for easy iteration
        predicted_boxes = boxes.tolist()
        predicted_classes = class_ids.tolist()

        # Save label files with normalized coordinates and class IDs
        save_labels(predicted_boxes, predicted_classes, frame_name, labels_dir, image_width, image_height)

        # Prepare SAM input
        predictor.set_image(image)
        predicted_boxes_tensor = torch.tensor(predicted_boxes).to(device)
        transformed_boxes = predictor.transform.apply_boxes_torch(predicted_boxes_tensor, image.shape[:2])

        # Perform segmentation for all transformed boxes
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

        # Convert all segmented masks to NumPy arrays and create an aggregate binary mask
        aggregate_mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # Loop through all masks to update the aggregate mask
        for mask in masks:
            mask_np = mask.cpu().numpy()

            # Ensure the mask is 2D by squeezing any unnecessary dimensions
            if mask_np.ndim > 2:
                mask_np = mask_np.squeeze()

            # Apply the mask to the aggregate mask using logical OR
            aggregate_mask = np.logical_or(aggregate_mask, mask_np > 0.5).astype(np.uint8)

        # Convert the aggregate mask to a binary mask with values 255 and 0
        binary_mask = np.where(aggregate_mask > 0, 255, 0).astype(np.uint8)

        white_background = np.ones_like(image) * 255

        new_image = np.where(aggregate_mask[..., np.newaxis], image, white_background).astype(np.uint8)

        # Save polygons in text format
        #save_polygons_txt(masks, frame_name, polygons_dir, image_width, image_height, predicted_classes)

        # Save binary masks as separate PNG files
        save_binary_masks(masks, frame_name, mask_dir, predicted_classes)

        # Plotting the results (optional, for visualization)
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(image)
        for mask, class_id in zip(masks, predicted_classes):
            show_mask(mask.cpu().numpy(), ax, random_color=True)
        for box, class_id in zip(predicted_boxes, predicted_classes):
            show_box(box, ax, class_id)
        plt.axis('off')

        # Original image path (save at actual size)
        original_image_path = os.path.join(original_images_dir, f"{frame_name}_original.png")
        cv2.imwrite(original_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        print(f"Original image saved at: {original_image_path}")

        # Save the masked frame with matplotlib (optional)
        output_frame_path = os.path.join(output_dir, f'{frame_name}.png')
        plt.savefig(output_frame_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        print(f"Saved masked frame: {output_frame_path}")

        # Saving binary masks
        output_image_path = os.path.join(mask_dir, f"{frame_name}_segmented.png")
        cv2.imwrite(output_image_path, binary_mask)
        print(f"Segmented image saved at: {output_image_path}")
    
    except Exception as e:
        print(f"Error processing frame {frame_name}: {e}")

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax, class_id):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    # Define a color map for classes (you can customize this)
    num_classes = len(coco_class_mapping)
    color_map = plt.cm.get_cmap('hsv', num_classes)
    color = color_map(int(class_id))[:3]
    class_name = coco_class_mapping.get(int(class_id), 'Unknown')
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor='none', lw=2))
    ax.text(x0, y0 - 5, f'{class_name}', color='white', fontsize=12, backgroundcolor='none')

if __name__ == "__main__":
    # Open the video file
    video_path = 'your_file.mp4'
    cap = cv2.VideoCapture(video_path)
    
    # Directories to save masked frames, binary masks, and labels
    output_dir = 'masked_frames'
    mask_dir = 'binary_masks'
    labels_dir = 'labels'
    polygons_dir = 'polygons'
    original_images_dir = 'original_images_dir'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(polygons_dir, exist_ok=True)
    os.makedirs(original_images_dir, exist_ok=True)

    # Initialize YOLO model
    model = YOLO('yolov8m.pt')  # Ensure this is trained for multiple classes
    model.conf = 0.3  # Set confidence threshold
    
    # Initialize SAM model
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    
    frame_count = 0
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        # Create a unique name for each frame
        frame_name = f'frame_{frame_count:04d}'

        # Process each frame and save outputs
        process_frame(frame, model, predictor, device, frame_name, output_dir, mask_dir, labels_dir, polygons_dir)
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
