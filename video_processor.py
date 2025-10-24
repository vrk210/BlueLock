import sys
from pyexpat import features

import cv2
import numpy as np
import torch
from fontTools.misc.arrayTools import normRect
from torchvision import models, transforms
from scipy.spatial.distance import cosine

# Pretrained model (e.g., ResNet18)
model = models.ResNet18(pretrained=True)
model.eval()

# Transformation for video frames
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_features(video_path):
    cap = cv2.VideoCapture(video_path)
    features = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = transform(frame).unsqueeze(0)

        with torch.no_grad():
            feature = model(input_tensor).numpy().flatten()
        features.append(feature)

    cap.release()
    return np.mean(features, axis=0)

def compare_videos(video1_path, video2_path):
    features1 = extract_features(video1_path)
    features2 = extract_features(video2_path)
    similarity = 1 - cosine(features1, features2)
    return similarity
    

