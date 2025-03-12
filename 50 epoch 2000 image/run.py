import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from generator import Generator  # Ensure this is the correct model file name

# =====================
# Load the Trained Model
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained generator model
model = Generator().to(device)
model.load_state_dict(torch.load("YGAN_generator.onnx", map_location=device))
model.eval()

# =====================
# Load and Preprocess Images
# =====================
def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    transform = transforms.ToTensor()
    return transform(image).unsqueeze(0).to(device)

left_image_path = "left_image.png"  # Change to your actual left image path
right_image_path = "right_image.png"  # Change to your actual right image path

left_tensor = load_image(left_image_path)
right_tensor = load_image(right_image_path)

# =====================
# Generate Disparity Map
# =====================
with torch.no_grad():
    disparity_map = model(left_tensor, right_tensor).cpu().squeeze(0).squeeze(0).numpy()

# Avoid division by zero for depth calculation
depth_map = 1.0 / (disparity_map + 1e-6)

# Resize maps to original image size
original_image = cv2.imread(left_image_path)
height, width = original_image.shape[:2]
disparity_map_resized = cv2.resize(disparity_map, (width, height), interpolation=cv2.INTER_LINEAR)
depth_map_resized = cv2.resize(depth_map, (width, height), interpolation=cv2.INTER_LINEAR)

# =====================
# Plot Disparity & Depth Maps
# =====================
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1); plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)); plt.title("Left Image")
plt.subplot(1, 3, 2); plt.imshow(disparity_map_resized, cmap='magma'); plt.title("Disparity Map")
plt.subplot(1, 3, 3); plt.imshow(depth_map_resized, cmap='inferno'); plt.title("Depth Map")
plt.show()
