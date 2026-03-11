import matplotlib

matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# Podaci iz terminala
loss_vrijednosti = [0.1995, 0.0274, 0.0181, 0.0100, 0.0059, 0.0026, 0.0010, 0.0013, 0.0056, 0.0018]
accuracy_vrijednosti = [100.0] * 10 

epochs = range(1, 11)

plt.figure(figsize=(12, 5))

# Graf Gubitka
plt.subplot(1, 2, 1)
plt.plot(epochs, loss_vrijednosti, 'r-o', label='Training Loss')
plt.title('Krivulja Gubitka (Loss)')
plt.xlabel('Epoha')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

# Graf Točnosti
plt.subplot(1, 2, 2)
plt.plot(epochs, accuracy_vrijednosti, 'b-o', label='Training Accuracy')
plt.title('Krivulja Točnosti (Accuracy)')
plt.xlabel('Epoha')
plt.ylabel('Točnost (%)')
plt.ylim(0, 110)
plt.grid(True)
plt.legend()

plt.tight_layout()

# SPREMANJE SLIKE
plt.savefig('grafovi_treninga.png')
print("Uspjeh! Grafovi su spremljeni kao datoteka 'grafovi_treninga.png' u tvom folderu.")