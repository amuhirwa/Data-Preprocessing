import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

IMAGE_DIR = BASE_DIR / 'assets/images'
FEATURES_CSV = BASE_DIR / 'data' /  'image_features.csv'
AUGMENTATIONS = ['original', 'rotated', 'flipped', 'grayscale']
EXPRESSIONS = ['neutral', 'smiling', 'surprised']

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Pretrained model for embeddings
model = models.resnet18(weights='DEFAULT')
model = torch.nn.Sequential(*(list(model.children())[:-1])) 
model.eval()
model.to(DEVICE)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_image_paths():
    image_paths = []
    for fname in os.listdir(IMAGE_DIR):
        if fname.lower().endswith(('.jpg', '.png', 'jpeg')):
            base = os.path.splitext(fname)[0]
            parts = base.split('_')
            if len(parts) < 2:
                continue  
            member = '_'.join(parts[:-1])
            expr = parts[-1]
            if expr not in EXPRESSIONS:
                continue
            img_path = os.path.join(IMAGE_DIR, fname)
            image_paths.append((member, expr, img_path))
    return image_paths

def display_image(img, title):
    # Save image instead of displaying (better for batch processing)
    print(f"Processing: {title}")
    # Uncomment the lines below if you want to actually display images
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.title(title)
    # plt.axis('off')
    # plt.show()

def augment_image(img):
    # Original
    aug_imgs = {'original': img}
    # Rotated (by 45 degrees)
    (h, w) = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), 45, 1.0)
    aug_imgs['rotated'] = cv2.warpAffine(img, M, (w, h))
    # Flipped (horizontal)
    aug_imgs['flipped'] = cv2.flip(img, 1)
    # Grayscale
    aug_imgs['grayscale'] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return aug_imgs

def extract_histogram(img):
    # If grayscale, img is 2D
    if len(img.shape) == 2:
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist = hist.flatten()
    else:
        # Color histogram (concatenate channels)
        hist = []
        for i in range(3):
            h = cv2.calcHist([img], [i], None, [256], [0, 256])
            hist.extend(h.flatten())
        hist = np.array(hist)
    return hist

def extract_embedding(img):
    # Convert to PIL Image if needed
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        embedding = model(input_tensor).cpu().numpy().flatten()
    return embedding

def main():
    image_paths = get_image_paths()
    features = []
    for member, expr, path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f'Could not read {path}')
            continue
        # Display original
        display_image(img, f'{member} - {expr} (original)')
        # Augment
        aug_imgs = augment_image(img)
        for aug_type, aug_img in aug_imgs.items():
            # Display augmented image
            display_image(aug_img if len(aug_img.shape)==3 else cv2.cvtColor(aug_img, cv2.COLOR_GRAY2BGR), f'{member} - {expr} ({aug_type})')
            # Extract features
            hist = extract_histogram(aug_img)
            emb = extract_embedding(aug_img)
            # Save features
            row = {
                'member': member,
                'expression': expr,
                'augmentation': aug_type,
                'histogram': hist.tolist(),
                'embedding': emb.tolist(),
                'image_path': path
            }
            features.append(row)
    df = pd.DataFrame(features)
    df.to_csv(FEATURES_CSV, index=False)
    print(f'Features saved to {FEATURES_CSV}')

if __name__ == '__main__':
    main() 