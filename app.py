import streamlit as st
import cv2
import numpy as np
import re
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from ultralytics import YOLO
from paddleocr import PaddleOCR

import asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())
    
# --------------------- Image Enhancement Functions ---------------------
def apply_clahe(image):
    """Apply CLAHE to enhance contrast"""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def apply_gaussian_sharpening(image):
    """Apply Gaussian sharpening to enhance edges"""
    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
    return sharpened

# --------------------- Dehazing Model Architecture ---------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return out

class ImprovedDehazingUNet(nn.Module):
    def __init__(self):
        super(ImprovedDehazingUNet, self).__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(64),
            ResidualBlock(64),
        )
        # ... (rest of model architecture remains unchanged)
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(128),
            ResidualBlock(128),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(256),
            ResidualBlock(256),
        )
        self.dec3 = nn.Sequential(
            nn.Conv2d(256 + 128, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(128),
            ResidualBlock(128),
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(64),
            ResidualBlock(64),
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        enc1_out = self.enc1(x)
        enc2_out = self.enc2(enc1_out)
        enc3_out = self.enc3(enc2_out)
        dec3_out = self.dec3(torch.cat([enc3_out, enc2_out], dim=1))
        dec2_out = self.dec2(torch.cat([dec3_out, enc1_out], dim=1))
        dec1_out = self.dec1(dec2_out)
        return dec1_out

# --------------------- Model Loading Functions ---------------------
@st.cache_resource(show_spinner=False)
def load_yolo_model():
    return YOLO("best.pt")

@st.cache_resource(show_spinner=False)
def load_paddleocr_reader():
    return PaddleOCR(use_angle_cls=True, lang="en")

@st.cache_resource(show_spinner=False)
def load_dehazing_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImprovedDehazingUNet().to(device)
    checkpoint = torch.load("dehazing_unet.pth", map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    return model

# --------------------- Image Processing Functions ---------------------
def process_image(image_pil, dehaze_model):
    """Apply the dehazing model to an image"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    image_tensor = transform(image_pil).unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        dehazed_tensor = dehaze_model(image_tensor)
    
    dehazed_np = (dehazed_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)  # Convert back to image
    dehazed_pil = Image.fromarray(np.transpose(dehazed_np, (1, 2, 0)))  # Convert to PIL image
    return dehazed_pil

def clean_text(text):
    """Clean and format the extracted text"""
    # Remove all non-alphanumeric characters and convert to uppercase
    clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
    
    # Regex for formats like AB12CD3456 or TN18D1866 (without hyphens)
    format1 = re.search(r"([A-Z]{2})(\d{2})([A-Z]{1,2})(\d{4})", clean_text)
    
    # Regex for formats like TN-18-D-1866 (with hyphens)
    format2 = re.search(r"([A-Z]{2})-(\d{2})-([A-Z]{1,2})-(\d{4})", text.upper())
    
    # If the text matches the expected format, return it
    if format1:
        return format1.group(0)
    elif format2:
        # Remove hyphens for consistency
        return format2.group(0).replace("-", "")
    else:
        # If the text doesn't match the expected format, correct 0/O
        # Ensure the corrected text follows the format (alphabet)(alphabet)(number)(number)(alphabet)(alphabet)(number)(number)(number)(number)
        corrected_text = []
        for i, char in enumerate(clean_text):
            if i < 2 or (4 <= i < 6):
                # Positions 0,1 (alphabets) and 4,5 (alphabets): Replace 0 with O
                if char == '0':
                    corrected_text.append('O')
                else:
                    corrected_text.append(char)
            elif 2 <= i < 4 or i >= 6:
                # Positions 2,3 (numbers) and 6,7,8,9 (numbers): Replace O with 0
                if char == 'O':
                    corrected_text.append('0')
                else:
                    corrected_text.append(char)
            else:
                corrected_text.append(char)
        
        # Join the corrected characters and check if it matches the format
        corrected_text = ''.join(corrected_text)
        if re.match(r"[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}", corrected_text):
            return corrected_text
        else:
            return None  # Return None if the corrected text still doesn't match the format

# --------------------- Main Application ---------------------

def main():
    # Load models
    yolo_model = load_yolo_model()
    ocr_reader = load_paddleocr_reader()
    dehaze_model = load_dehazing_model()

    # Streamlit UI
    st.title("License Plate Detection & OCR with Dehazing")
    st.write("Upload an image to detect license plate, enhance it, and read text.")

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read and process image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Display original image
        st.image(image_rgb, caption="Original Image", use_container_width=True)

        # First attempt: Directly run YOLO and PaddleOCR
        with st.spinner("Detecting license plate without preprocessing..."):
            results = yolo_model.predict(image_rgb)
            found_any_plate = False
            plate_texts = []

            for result in results:
                boxes = result.boxes
                if len(boxes) == 0:
                    continue

                best_box = max(boxes, key=lambda b: b.conf[0])
                found_any_plate = True
                
                # Crop plate region from original image
                x1, y1, x2, y2 = map(int, best_box.xyxy[0])
                cropped_plate = image_rgb[y1:y2, x1:x2]
                
                # Display cropped plate
                st.image(cropped_plate, caption="Cropped License Plate (No Preprocessing)", use_container_width=True)

                # OCR processing
                with st.spinner("Reading license plate text..."):
                    ocr_result = ocr_reader.ocr(cropped_plate, cls=True)
                    extracted_text = " ".join([word_info[1][0] for line in ocr_result for word_info in line])
                    cleaned_text = clean_text(extracted_text)
                    
                    if cleaned_text:
                        plate_texts.append(cleaned_text)

            # Display results from first attempt
            if found_any_plate:
                if plate_texts:
                    st.subheader("Detected License Plate Text (No Preprocessing):")
                    for idx, text in enumerate(plate_texts, 1):
                        st.success(f"Plate {idx}: {text}")
                    return  # Exit if successful
                else:
                    st.warning("License plate detected but text recognition failed. Trying with preprocessing and dehazing...")
            else:
                st.warning("No license plate detected. Trying with preprocessing and dehazing...")

        # Fallback to preprocessing and dehazing
        with st.spinner("Enhancing image quality with preprocessing and dehazing..."):
            try:
                # Apply CLAHE and Gaussian sharpening
                enhanced_image = apply_clahe(image_rgb)
                enhanced_image = apply_gaussian_sharpening(enhanced_image)
                
                # Dehaze the enhanced image
                image_pil = Image.fromarray(enhanced_image)
                dehazed_pil = process_image(image_pil, dehaze_model)
                dehazed_image = np.array(dehazed_pil)
                
                # Display enhanced and dehazed image
                st.image(dehazed_image, caption="Enhanced and Dehazed Image", use_container_width=True)

                # Run YOLO and PaddleOCR on the enhanced image
                with st.spinner("Detecting license plate with preprocessing and dehazing..."):
                    results = yolo_model.predict(dehazed_image)

                found_any_plate = False
                plate_texts = []

                for result in results:
                    boxes = result.boxes
                    if len(boxes) == 0:
                        continue

                    best_box = max(boxes, key=lambda b: b.conf[0])
                    found_any_plate = True
                    
                    # Crop plate region from enhanced image
                    x1, y1, x2, y2 = map(int, best_box.xyxy[0])
                    cropped_plate = dehazed_image[y1:y2, x1:x2]
                    
                    # Display cropped plate
                    st.image(cropped_plate, caption="Cropped License Plate (With Preprocessing)", use_container_width=True)

                    # OCR processing
                    with st.spinner("Reading license plate text..."):
                        ocr_result = ocr_reader.ocr(cropped_plate, cls=True)
                        extracted_text = " ".join([word_info[1][0] for line in ocr_result for word_info in line])
                        cleaned_text = clean_text(extracted_text)
                        
                        if cleaned_text:
                            plate_texts.append(cleaned_text)

                # Display results from fallback attempt
                if found_any_plate:
                    if plate_texts:
                        st.subheader("Detected License Plate Text (With Preprocessing):")
                        for idx, text in enumerate(plate_texts, 1):
                            st.success(f"Plate {idx}: {text}")
                    else:
                        st.warning("License plate detected but text recognition failed even with preprocessing and dehazing")
                else:
                    st.error("No license plate detected even with preprocessing and dehazing. Please try another image.")
            except Exception as e:
                st.error(f"Preprocessing and dehazing failed: {str(e)}")

if __name__ == "__main__":
    main()
