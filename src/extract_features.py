from torchvision import models, transforms
import torch
import numpy as np
import cv2
import os

model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove classification layer
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def extract_embeddings(img_dir, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    embeddings = []
    filenames = []

    for file in os.listdir(img_dir):
        img_path = os.path.join(img_dir, file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            feat = model(tensor).squeeze().numpy()
        embeddings.append(feat)
        filenames.append(file)

    np.save(output_path, {"embeddings": embeddings, "files": filenames})
