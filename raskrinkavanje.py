import matplotlib
matplotlib.use('Agg') # Forsiramo spremanje u datoteku bez otvaranja prozora
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import cv2
import numpy as np
import os
from torchvision import models, transforms
from PIL import Image

# 1. Postavke i učitavanje modela
device = torch.device("cpu")
model = models.efficientnet_b0()
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, 2)

# Učitavanje tvog istreniranog modela
model.load_state_dict(torch.load('overfitted_model.pth', map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def generiraj_i_spremi_gradcam(putanja_slike, stvarna_klasa, redni_broj):
    img_pil = Image.open(putanja_slike).convert('RGB')
    img_tensor = transform(img_pil).unsqueeze(0)

    features = []
    def hook_feature(module, input, output):
        features.append(output)

    handle = model.features[8].register_forward_hook(hook_feature)

    output = model(img_tensor)
    pred_idx = output.argmax(dim=1).item()
    klase_nazivi = ["ZDRAVO DIJETE", "PNEUMONIJA (ODRASLI)"]

    feature_map = features[0].detach().numpy()[0]
    weights = np.mean(feature_map, axis=(1, 2))
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-10)
    
    handle.remove()

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    img_res = np.array(img_pil.resize((224, 224)))
    superimposed_img = (heatmap * 0.4 + img_res * 0.6).astype(np.uint8)

    # CRTANJE
    plt.figure(figsize=(12, 7))
    plt.subplot(1, 2, 1)
    plt.imshow(img_res)
    plt.title(f"Originalni Rendgen\nStvarno: {stvarna_klasa}")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(superimposed_img)
    plt.title(f"Grad-CAM Fokus\nPredviđeno: {klase_nazivi[pred_idx]}")
    plt.axis('off')

    # SPREMANJE (umjesto show)
    ime_izlazne_slike = f"ANALIZA_{stvarna_klasa}_{redni_broj}.png"
    plt.savefig(ime_izlazne_slike)
    plt.close() # Zatvaramo sliku da oslobodimo memoriju
    print(f"Spremljeno: {ime_izlazne_slike}")

# 3. Glavna petlja
podaci_path = 'skup_podataka'
klase_folderi = ['NORMAL', 'PNEUMONIA']
broj_slika_po_klasi = 3 

print("Započinjem generiranje Grad-CAM slika...")

for kl in klase_folderi:
    putanja_do_foldera = os.path.join(podaci_path, kl)
    sve_slike = [f for f in os.listdir(putanja_do_foldera) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for i, ime_slike in enumerate(sve_slike[:broj_slika_po_klasi]):
        puna_putanja = os.path.join(putanja_do_foldera, ime_slike)
        generiraj_i_spremi_gradcam(puna_putanja, kl, i+1)

print("\nGotovo! Pogledaj u folderu OSIRV slike koje počinju s 'ANALIZA_'.")