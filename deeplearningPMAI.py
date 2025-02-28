import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import cv2
import numpy as np
from einops import rearrange
from torchvision.models.video import r3d_18  # ResNet-3D backbone
from transformers import TimeSformerModel

# Optical Flow Extraction using Dense Optical Flow (Farneback)
def extract_optical_flow(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    flow_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_data.append(flow)
        prev_gray = gray

    cap.release()
    return np.array(flow_data)

# 3D CNN Backbone
class PitchingMovementModel(nn.Module):
    def __init__(self):
        super(PitchingMovementModel, self).__init__()
        self.backbone = r3d_18(pretrained=True)
        self.backbone.fc = nn.Identity()  # Remove final classifier layer
        self.temporal_model = TimeSformerModel.from_pretrained("facebook/timesformer-base-finetuned-k400")
        self.fc = nn.Linear(768, 2)  # Binary classification: Normal vs Anomalous

    def forward(self, x):
        x = self.backbone(x)  # Extract spatiotemporal features
        x = rearrange(x, "b c t h w -> b t (c h w)")  # Reshape for transformer
        x = self.temporal_model(x).last_hidden_state[:, 0, :]  # Extract temporal embeddings
        return self.fc(x)

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PitchingMovementModel().to(device)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training Function
def train_model(dataloader, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        for batch in dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Anomaly Detection with Autoencoder
class PitchAnomalyDetector(nn.Module):
    def __init__(self):
        super(PitchAnomalyDetector, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Compute Anomaly Score
def compute_anomaly_score(original, reconstructed):
    return torch.mean((original - reconstructed) ** 2)

# Usage Example
video_path = "pitcher_highspeed.mp4"
flow_data = extract_optical_flow(video_path)
input_tensor = torch.tensor(flow_data, dtype=torch.float32).unsqueeze(0).to(device)
anomaly_detector = PitchAnomalyDetector().to(device)

with torch.no_grad():
    reconstructed = anomaly_detector(input_tensor)
    score = compute_anomaly_score(input_tensor, reconstructed)
    print(f"Anomaly Score: {score.item()}")
