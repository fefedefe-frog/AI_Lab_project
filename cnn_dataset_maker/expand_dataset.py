import os
import cv2
import csv
import random
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np


# === CONFIG ===
INPUT_FOLDER = "./simple_card_images"
OUTPUT_FOLDER = "output_cut_simple_cards"
CSV_PATH = "./output/dataset.csv"
NUM_AUGMENTAZIONI = 25
DIMENSIONE_CROP_NUM = (190, 90)  # Altezza x Larghezza crop angolo superiore sinistro
DIMENSIONE_CROP_FIG = (160, 65)

semi_mappa = {
    "clubs": "fiori",      # fiori
    "diamonds": "quadri",   # quadri
    "hearts": "cuori",     # cuori
    "spades": "picche"      # picche
}

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# === Augmentazioni leggere da applicare randomicamente ===
def random_augment(image):
    transformazioni = transforms.Compose([
        transforms.RandomAdjustSharpness(sharpness_factor=2),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0)),
    ])
    return transformazioni(image)

# === Apre o crea CSV ===
file_esiste = os.path.exists(CSV_PATH)
csv_file = open(CSV_PATH, mode='a', newline='')
csv_writer = csv.writer(csv_file)
if not file_esiste:
    csv_writer.writerow(['path', 'seme', 'numero'])

# === Loop immagini originali ===
counter = 0
for nome_file in tqdm(sorted(os.listdir(INPUT_FOLDER))):
    if not nome_file.endswith((".jpg", ".jpeg", ".png")):
        continue

    path_img = os.path.join(INPUT_FOLDER, nome_file)
    pil_img_orig = Image.open(path_img).convert("RGB")
    base_nome = os.path.splitext(nome_file)[0]  # es: "2_of_clubs"

    try:
        numero_str, _, seme_str = base_nome.split("_")
        seme_label = semi_mappa[seme_str.lower()]
        numero_label = numero_str.upper()
    except Exception as e:
        print(f"Errore parsing nome file: {nome_file}")
        continue

    for i in range(NUM_AUGMENTAZIONI):
        # Applica trasformazioni
        pil_aug = random_augment(pil_img_orig)

        # Converti in numpy per cropping
        img_aug_np = cv2.cvtColor(np.array(pil_aug), cv2.COLOR_RGB2BGR)

        # Crop angolo in alto a sinistra (sperimentale, si può modificare)
        if(numero_label in ["J","Q","K"]):
          crop = img_aug_np[3:3 + DIMENSIONE_CROP_FIG[0], 5:5 + DIMENSIONE_CROP_FIG[1]]  # [y1:y2, x1:x2]
        else:
          crop = img_aug_np[5:5 + DIMENSIONE_CROP_NUM[0], 5:5 + DIMENSIONE_CROP_NUM[1]]  # [y1:y2, x1:x2]

        # Salva immagine
        nome_output = f"{counter}.jpg"
        path_output = os.path.join(OUTPUT_FOLDER, nome_output)
        cv2.imwrite(path_output, crop)

        # Scrivi riga CSV
        csv_writer.writerow([f"{OUTPUT_FOLDER}\{nome_output}",seme_label,numero_label])
        counter += 1

csv_file.close()
print(f"\n✅ Dataset esteso completato. Totale immagini generate: {counter}")
