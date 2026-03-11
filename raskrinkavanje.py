import torch
import torch.nn as nn
import cv2
import numpy as np
import os

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

from torchvision import models, transforms
from PIL import Image

device = torch.device("cpu")
model = models.efficientnet_b0()
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, 2)

# Učitavanje modela
model.load_state_dict(torch.load('overfitted_model.pth'))
model.eval()

# Odabiranje same slike
putanja_slike = 'PNEUMONIA/odrasli_0.jpg' 

if not os.path.exists(putanja_slike):
    # Ako nema slike 0, pokušaj naći bilo koju drugu u tom folderu
    datoteke = os.listdir('PNEUMONIA')
    if datoteke:
        putanja_slike = os.path.join('PNEUMONIA', datoteke[0])
    else:
        print("Greška: Folder PNEUMONIA je prazan!")
        exit()

img_pil = Image.open(putanja_slike).convert('RGB')

# Priprema slike
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
img_tensor = transform(img_pil).unsqueeze(0)

# Grad-CAM logika da bi sada on usporedio obični rendgen i Grad-CAM
features = []
def hook_feature(module, input, output):
    features.append(output)

model.features[8].register_forward_hook(hook_feature)
output = model(img_tensor)
target_class = output.argmax(dim=1).item()
klase = ["NORMALNO (DIJETE)", "PNEUMONIJA (ODRASLI)"]

feature_map = features[0].detach().numpy()[0]
weights = np.mean(feature_map, axis=(1, 2))
cam = np.zeros(feature_map.shape[1:], dtype=np.float32)

for i, w in enumerate(weights):
    cam += w * feature_map[i, :, :]

cam = np.maximum(cam, 0)
cam = cv2.resize(cam, (224, 224))
cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-10)

# SPREMANJE REZULTATA
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
img_res = np.array(img_pil.resize((224, 224)))
superimposed_img = (heatmap * 0.4 + img_res * 0.6).astype(np.uint8)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(img_res)
plt.title("Originalni Rendgen")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(superimposed_img)
plt.title(f"AI Vizualizacija\nPredviđeno: {klase[target_class]}")
plt.axis('off')

plt.savefig('DOKAZ_OVERFITTINGA.png')
print("Uspjeh! Pogledaj datoteku 'DOKAZ_OVERFITTINGA.png' u svom folderu.")