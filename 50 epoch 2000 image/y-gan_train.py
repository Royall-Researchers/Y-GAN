import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==============================
# Dataset Loader for Stereo Images
# ==============================
class StereoDataset(Dataset):
    def __init__(self, left_folder, right_folder, transform=None):
        self.left_folder = left_folder
        self.right_folder = right_folder
        self.left_images = sorted(os.listdir(left_folder))
        self.right_images = sorted(os.listdir(right_folder))
        self.transform = transform

    def __len__(self):
        return len(self.left_images)

    def __getitem__(self, idx):
        left_path = os.path.join(self.left_folder, self.left_images[idx])
        right_path = os.path.join(self.right_folder, self.right_images[idx])

        left_image = cv2.imread(left_path)
        right_image = cv2.imread(right_path)

        # Convert to RGB and normalize
        left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        if self.transform:
            left_image = self.transform(left_image)
            right_image = self.transform(right_image)

        return left_image, right_image

# ==============================
# Y-GAN Generator
# ==============================
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # Output disparity map
        )

    def forward(self, left_image, right_image):
        x = torch.cat([left_image, right_image], dim=1)  # Concatenate stereo images
        x = self.encoder(x)
        disparity_map = self.decoder(x)
        return disparity_map

# ==============================
# Discriminator Model
# ==============================
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((8, 8))  # Reduce feature map size
        self.fc = nn.Linear(256 * 8 * 8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return self.sigmoid(x)

# ==============================
# Training Setup
# ==============================
criterion_BCE = nn.BCELoss()
criterion_L1 = nn.L1Loss()

# Load Dataset
transform = transforms.ToTensor()
dataset = StereoDataset(
    left_folder="/content/drive/MyDrive/Colab Notebooks/image_0",   # ✅ Change to your dataset folder path
    right_folder="/content/drive/MyDrive/Colab Notebooks/image_1", # ✅ Change to your dataset folder path
    transform=transform
)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)  # Train with batch size 4

# Initialize Models
generator = Generator().to(device)
discriminator = Discriminator().to(device)

optimizer_G = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

# ==============================
# Training Loop
# ==============================
num_epochs = 50  # Train for 50 epochs

for epoch in range(num_epochs):
    for left_tensor, right_tensor in dataloader:
        left_tensor, right_tensor = left_tensor.to(device), right_tensor.to(device)

        # Train Discriminator
        optimizer_D.zero_grad()

        fake_disparity = generator(left_tensor, right_tensor)
        real_disparity = torch.rand_like(fake_disparity).to(device)  # Replace with real data

        fake_labels = torch.zeros(fake_disparity.size(0), 1).to(device)
        real_labels = torch.ones(real_disparity.size(0), 1).to(device)

        fake_loss = criterion_BCE(discriminator(fake_disparity.detach()), fake_labels)
        real_loss = criterion_BCE(discriminator(real_disparity), real_labels)
        d_loss = fake_loss + real_loss

        d_loss.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()

        g_loss = criterion_BCE(discriminator(fake_disparity), real_labels) + 10 * criterion_L1(fake_disparity, real_disparity)
        g_loss.backward()
        optimizer_G.step()

    print(f"Epoch [{epoch+1}/{num_epochs}] - D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

    # Save Model Every 10 Epochs
    if epoch % 10 == 0:
        torch.save(generator.state_dict(), f"generator_epoch_{epoch}.pth")
        torch.save(discriminator.state_dict(), f"discriminator_epoch_{epoch}.pth")

# ==============================
# Export both Generator and Discriminator to ONNX format
# ==============================
generator.eval()
discriminator.eval()

dummy_left = torch.randn(1, 3, 256, 256, device=device)
dummy_right = torch.randn(1, 3, 256, 256, device=device)

# Export Generator to ONNX
torch.onnx.export(generator, (dummy_left, dummy_right), "YGAN_generator.onnx", export_params=True, opset_version=11)
print("✅ Generator model exported to ONNX format successfully!")

# Export Discriminator to ONNX
dummy_disparity = torch.randn(1, 1, 256, 256, device=device)  # Example input for discriminator
torch.onnx.export(discriminator, dummy_disparity, "YGAN_discriminator.onnx", export_params=True, opset_version=11)
print("✅ Discriminator model exported to ONNX format successfully!")

# ==============================
# Visualizing Disparity & Depth Maps
# ==============================
def plot_disparity_depth(left_img, right_img, generator):
    # Ensure the input images are in the correct format and the same size
    transform = transforms.ToTensor()
    left_tensor, right_tensor = transform(left_img).unsqueeze(0).to(device), transform(right_img).unsqueeze(0).to(device)

    with torch.no_grad():
        # Generate the disparity map
        disparity_map = generator(left_tensor, right_tensor).cpu().squeeze(0).squeeze(0).numpy()

    # Depth map calculation (avoid division by zero errors)
    depth_map = 1.0 / (disparity_map + 1e-6)

    # Resize the disparity and depth maps to match the input image size
    disparity_map_resized = cv2.resize(disparity_map, (left_img.shape[1], left_img.shape[0]), interpolation=cv2.INTER_LINEAR)
    depth_map_resized = cv2.resize(depth_map, (left_img.shape[1], left_img.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Plotting the results
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1); plt.imshow(left_img); plt.title("Left Image")
    plt.subplot(1, 3, 2); plt.imshow(disparity_map_resized, cmap='magma'); plt.title("Disparity Map")
    plt.subplot(1, 3, 3); plt.imshow(depth_map_resized, cmap='inferno'); plt.title("Depth Map")
    plt.show()
