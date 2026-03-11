import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import os
from PIL import Image

# Postavke putanja
putanja_dijete = 'NORMAL/djeca_0.jpg'     
putanja_odrasli = 'PNEUMONIA/odrasli_0.jpg' 

def dohvati_bilo_koju_sliku(folder):
    if not os.path.exists(folder): return None
    datoteke = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    return os.path.join(folder, datoteke[0]) if datoteke else None

if not os.path.exists(putanja_dijete): putanja_dijete = dohvati_bilo_koju_sliku('NORMAL')
if not os.path.exists(putanja_odrasli): putanja_odrasli = dohvati_bilo_koju_sliku('PNEUMONIA')

# Učitavanje i skaliranje na istu visinu radi bolje usporedbe
img_dijete = Image.open(putanja_dijete).convert('L')
img_odrasli = Image.open(putanja_odrasli).convert('L')

# Kreiranje čistog usporednog prikaza
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].imshow(img_dijete, cmap='gray')
ax[0].set_title("Klasa: NORMAL (Pedijatrijski pacijent)")
ax[0].axis('off')

ax[1].imshow(img_odrasli, cmap='gray')
ax[1].set_title("Klasa: PNEUMONIA (Odrasli pacijent)")
ax[1].axis('off')

plt.tight_layout()
plt.savefig('usporedba_anatomije.png', dpi=300)
print("Uspjeh! Slika 'usporedba_anatomije.png' je generirana bez teksta.")