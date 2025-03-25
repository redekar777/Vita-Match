import streamlit as st
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from skimage.color import rgb2lab
from PIL import Image, ImageDraw, ImageFont
import time

# Load YOLO model
model = YOLO("best.pt")  # Replace with your trained model

# Vita shade reference dictionary
VITA_SHADES = {  
    "0M1": [70.9, -0.4, 1.6], "0M2": [67.80, -0.2, 2.8], "0M3": [66.8, -0.4, 3.3],
    "1M1": [64.7, -0.1, 5.4], "1M2": [65.00, 1, 8.7], "2L1.5": [61.8, 0.2, 8.7],
    "2M1": [60.5, 0.3, 9.1], "2M2": [59.1, 0.5, 10.2], "2M3": [58.0, 0.6, 11.1],
    "3L1.5": [57.4, 0.8, 12.0], "3M1": [56.8, 0.9, 12.4], "3M2": [55.5, 1.1, 13.0],
    "3M3": [54.4, 1.3, 13.6], "4L1.5": [53.7, 1.5, 14.2], "4M1": [52.9, 1.7, 14.6],
    "4M2": [51.8, 1.9, 15.2], "4M3": [50.7, 2.1, 15.8], "5M1": [49.6, 2.3, 16.4],
    "5M2": [48.5, 2.5, 17.0], "5M3": [47.3, 2.7, 17.6]
}

IOU_THRESHOLD = 0.4

def find_closest_vita_shade(lab_value):
    return min(VITA_SHADES, key=lambda shade: np.linalg.norm(np.array(lab_value) - np.array(VITA_SHADES[shade])))

def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2
    xi1, yi1 = max(x1, x1_p), max(y1, y1_p)
    xi2, yi2 = min(x2, x2_p), min(y2, y2_p)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union_area = (x2 - x1) * (y2 - y1) + (x2_p - x1_p) * (y2_p - y1_p) - inter_area
    return inter_area / union_area if union_area > 0 else 0

def process_image(image):
    image_rgb = np.array(image)
    results = model(image_rgb)
    
    filtered_boxes = []
    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, box[:4])
        if all(calculate_iou([x1, y1, x2, y2], prev) <= IOU_THRESHOLD for prev in filtered_boxes):
            filtered_boxes.append([x1, y1, x2, y2])
    
    total_teeth = len(filtered_boxes)
    progress_bar = st.progress(0)
    
    image_draw = image.copy()
    draw = ImageDraw.Draw(image_draw)
    
    for x1, y1, x2, y2 in filtered_boxes:
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    
    highlighted_images = []
    
    for idx, (x1, y1, x2, y2) in enumerate(filtered_boxes):
        tooth_crop = image_rgb[y1:y2, x1:x2]
        tooth_lab = rgb2lab(tooth_crop).mean(axis=(0, 1))
        closest_shade = find_closest_vita_shade(tooth_lab)
        
        highlight_image = image.copy()
        highlight_draw = ImageDraw.Draw(highlight_image)
        highlight_draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        
        label_bg_color = (64, 224, 208)  # Turquoise blue
        label_text_color = "white"
        label_height = 30
        
        highlight_draw.rectangle([(x1, y2), (x2, y2 + label_height)], fill=label_bg_color)
        highlight_draw.text((x1 + 5, y2 + 5), f"Tooth {idx+1}: {closest_shade}", fill=label_text_color)
        
        highlighted_images.append(highlight_image)
        
        progress_bar.progress(int(((idx + 1) / total_teeth) * 100))
        time.sleep(0.1)
    
    progress_bar.empty()
    return total_teeth, image_draw, highlighted_images

st.title("Vita Match")
# st.image("logo.png", width=150)
option = st.radio("Choose Input Method:", ["Upload Image", "Take Photo"], index=0)

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
elif option == "Take Photo":
    uploaded_file = st.camera_input("Take a photo")

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    teeth_count, detected_image, highlighted_images = process_image(image)
    
    st.subheader(f"Total Detected Teeth: {teeth_count}")
    st.image(detected_image, caption="Teeth Detection Result", use_container_width=True)
    
    for img in highlighted_images:
        st.image(img, caption="Highlighted Tooth", use_container_width=True)
