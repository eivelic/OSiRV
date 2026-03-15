import matplotlib
matplotlib.use('Agg')  # Rješava _tkinter.TclError tako da samo sprema sliku bez otvaranja prozora
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import os

# 1. Postavke
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = 'skup_podataka'  # Folder u kojem su NORMAL i PNEUMONIA
batch_size = 16
epochs = 10

print(f"Započinjem rad. Uređaj: {device}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 2. Učitavanje i podjela (70% Trening, 15% Validacija, 15% Test)
if not os.path.exists(data_dir):
    print(f"GREŠKA: Ne mogu pronaći folder '{data_dir}'. Provjeri jesi li ga napravio.")
    exit()

full_dataset = datasets.ImageFolder(data_dir, transform=transform)

train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_data, val_data, test_data = random_split(full_dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# 3. Model (EfficientNet-B0)
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

# 4. Trening i Validacija
print("Započinjem treniranje...")
for epoch in range(epochs):
    model.train()
    train_loss, train_correct = 0.0, 0
    for images, labels in tqdm(train_loader, desc=f"Epoha {epoch+1}/{epochs}"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        train_correct += torch.sum(preds == labels.data)

    # Validacija
    model.eval()
    val_loss, val_correct = 0.0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += torch.sum(preds == labels.data)

    history['train_loss'].append(train_loss / train_size)
    history['val_loss'].append(val_loss / val_size)
    history['train_acc'].append(train_correct.double().item() / train_size)
    history['val_acc'].append(val_correct.double().item() / val_size)
    
    print(f"Loss: {history['train_loss'][-1]:.4f} | Acc: {history['train_acc'][-1]:.4f}")

# 5. Testiranje (Metrike koje profesor traži)
print("\nEvaluacija na neviđenom testnom skupu...")
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

print("\n--- IZVJEŠTAJ KLASIFIKACIJE ---")
print(classification_report(y_true, y_pred, target_names=full_dataset.classes))

# 6. Generiranje i spremanje grafova
print("\nGeneriram grafove...")
plt.figure(figsize=(12, 5))

# Graf gubitka (Loss)
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Trening')
plt.plot(history['val_loss'], label='Validacija')
plt.title('Krivulja gubitka (Loss)')
plt.xlabel('Epoha')
plt.ylabel('Gubitak')
plt.legend()

# Graf točnosti (Accuracy)
plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Trening')
plt.plot(history['val_acc'], label='Validacija')
plt.title('Krivulja točnosti (Accuracy)')
plt.xlabel('Epoha')
plt.ylabel('Točnost')
plt.legend()

plt.tight_layout()
plt.savefig('krivulje_treniranja.png')

# 7. Spremanje modela
torch.save(model.state_dict(), 'overfitted_model.pth')
print("\nUspjeh! Model je spremljen, a grafovi se nalaze u 'krivulje_treniranja.png'.")