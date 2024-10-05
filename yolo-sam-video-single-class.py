import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
import os

def save_polygons_txt(masks, frame_name, polygons_dir, image_width, image_height):
    """
    Save mask contours as polygons in text format with normalized coordinates.
    """
    polygons = []
    polygon_file_path = os.path.join(polygons_dir, f'{frame_name}_polygons.txt')

    with open(polygon_file_path, 'w') as file:
        for i, mask in enumerate(masks):
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
                # Write each polygon as a new line in the text file
                polygon_str = ' '.join([f'{x} {y}' for x, y in normalized_polygon])
                file.write(polygon_str + '\n')
    
    print(f"Saved polygons in text format: {polygon_file_path}")

def save_binary_masks(masks, frame_name, mask_dir):
    """
    Save each individual binary mask as a separate PNG file.
    """
    for i, mask in enumerate(masks):
        mask_np = mask.cpu().numpy().astype(np.uint8) * 255  # Convert mask to binary image
        mask_path = os.path.join(mask_dir, f'{frame_name}_mask_{i}.png')
        cv2.imwrite(mask_path, mask_np)
        print(f"Saved binary mask: {mask_path}")

def save_labels(predicted_boxes, frame_name, labels_dir, image_width, image_height):
    """
    Save bounding boxes in a text file with the same name as the frame using normalized coordinates.
    """
    label_path = os.path.join(labels_dir, f'{frame_name}.txt')
    
    with open(label_path, 'w') as f:
        for box in predicted_boxes:
            x0, y0, x1, y1 = box
            x_center = ((x0 + x1) / 2) / image_width
            y_center = ((y0 + y1) / 2) / image_height
            width = (x1 - x0) / image_width
            height = (y1 - y0) / image_height
            
            # Assuming class_id is 0 for simplicity
            f.write(f"0 {x_center} {y_center} {width} {height}\n")

def process_frame(frame, model, predictor, device, frame_name, output_dir, mask_dir, labels_dir, polygons_dir):
    try:
        # Convert frame to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_height, image_width = image.shape[:2]

        # YOLO object detection
        with torch.no_grad():
            results = model.predict(source=image, conf=0.20, device=device)
        
        # Get bounding boxes
        predicted_boxes = results[0].boxes.xyxy.tolist()
        if len(predicted_boxes) == 0:
            print(f"No objects detected in frame: {frame_name}")
            return
        
        predicted_boxes = np.array(predicted_boxes)
        if predicted_boxes.ndim == 1:
            predicted_boxes = predicted_boxes[np.newaxis, :]
        
        # Save label files with normalized coordinates
        save_labels(predicted_boxes, frame_name, labels_dir, image_width, image_height)
        
        # Prepare SAM input
        predictor.set_image(image)
        predicted_boxes = torch.from_numpy(predicted_boxes).to(device)
        transformed_boxes = predictor.transform.apply_boxes_torch(predicted_boxes, image.shape[:2])

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

        white_background=np.ones_like(image) * 255

        new_image = np.where(aggregate_mask[..., np.newaxis], image, white_background).astype(np.uint8)

        # Save polygons in text format
        #save_polygons_txt(masks, frame_name, polygons_dir, image_width, image_height)

        # Save binary masks as separate PNG files
        save_binary_masks(masks, frame_name, mask_dir)

        # Plotting the results (optional, for visualization)
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(image)
        for mask in masks:
            show_mask(mask.cpu().numpy(), ax, random_color=True)
        for box in predicted_boxes:
            show_box(box.cpu().numpy(), ax)
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
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

if __name__ == "__main__":
    # Open the video file
    video_path = 'your_video.mp4'
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
    model = YOLO('best.pt')
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
