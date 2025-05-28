import csv

import cv2
import os
import random
import numpy as np

# Percorsi
INPUT_DIR_PATH = "input_images"
OUTPUT_DIR_PATH = "output"
CSV_PATH = "output/dataset.csv"

img_num_for_card= 50

# Lista file
card_folders = [folder for folder in os.listdir(INPUT_DIR_PATH)]

os.makedirs(OUTPUT_DIR_PATH, exist_ok=True)
os.makedirs(os.path.dirname(os.path.join(OUTPUT_DIR_PATH, CSV_PATH)), exist_ok=True)


with open(CSV_PATH, "w", newline="") as csvfile:
  csv_writer = csv.writer(csvfile)
  csv_writer.writerow(["image_path", "seme", "numero"])

  for folder in card_folders:

    numero, seme= folder.split(".")
    print(f"generating card dataset for: {numero} of {seme}")


    folder_path = os.path.join(INPUT_DIR_PATH, folder)
    card_output_dir_path= os.path.join(OUTPUT_DIR_PATH, folder)

    os.makedirs(card_output_dir_path, exist_ok=True)
    images= [img for img in os.listdir(folder_path) if img.endswith(".jpg")]
    for num, image in enumerate(images):
      num+= 1
      if num > img_num_for_card:
        break

      # Viene caricata la carta
      card_path = os.path.join(folder_path, image)
      card_rgba = cv2.imread(card_path, cv2.IMREAD_UNCHANGED) # Carico anche alpha channel per la trasparenza
      if card_rgba is None:
        print(f"⚠️ Immagine non caricata correttamente: {card_path}, salto...")
        continue

      # Viene ridimensionata la carta a 200x200 (anche se in teoria sono già tutte di questa dimensione)
      card_rgba = cv2.resize(card_rgba, (200, 200))

      # Vengono separati i canali
      card_rgb = card_rgba[:,:,:3] # primi 3 canali

      # Ritaglio la parte di immagine che interessa a me, ovvero solo l'angolo in alto a sinistra contentente simbolo e numero
      card_cropped= card_rgb[0:80, 0:60]

      # Vengono applicate le rotazioni
      '''angle = random.randint(-20,20)
      center = (w // 2, h // 2)
      rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
      card_rgb = cv2.warpAffine(card_rgb, rot_matrix, (w, h), flags=cv2.INTER_LINEAR)
      alpha = cv2.warpAffine(alpha, rot_matrix, (w, h), flags=cv2.INTER_LINEAR)'''

      # Vengono applicate modifiche di luminosità casuali
      brightness = random.uniform(0.8, 1.2)
      card_cropped = np.clip(card_cropped * brightness, 0, 255).astype(np.uint8)


      # Viene salvata l'immagine
      img_name = f"{num}.jpg"
      card_output_full_path = os.path.join(card_output_dir_path, img_name)
      cv2.imwrite(card_output_full_path, card_cropped)

      # Aggiorno il csv
      csv_writer.writerow([card_output_full_path, seme, numero])


print("✅ Dataset creato con OpenCV!")