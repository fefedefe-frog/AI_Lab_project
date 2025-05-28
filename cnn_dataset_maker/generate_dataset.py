import csv
import os
import sys

import cv2
import random
import numpy as np

from cli_utilities.simple_progress_bar import ProgressBar

# Percorsi
SCRIPT_DIR= os.path.dirname(os.path.realpath(__file__))
INPUT_DIR_PATH = f"{SCRIPT_DIR}/input_images"
OUTPUT_DIR_PATH = f"{SCRIPT_DIR}/output"
CSV_PATH = "dataset.csv"

img_num_for_card= 50

# Lista file
card_folders = [folder for folder in os.listdir(INPUT_DIR_PATH)]

os.makedirs(OUTPUT_DIR_PATH, exist_ok=True)
os.makedirs(os.path.dirname(os.path.join(OUTPUT_DIR_PATH, CSV_PATH)), exist_ok=True)

# Grafichina per il terminale
ProgressBar= ProgressBar()

with open(os.path.join(OUTPUT_DIR_PATH, CSV_PATH), "w", newline="") as csvfile:
  csv_writer = csv.writer(csvfile)
  csv_writer.writerow(["image_path", "seme", "numero"])

  for card_num, folder in enumerate(card_folders):
    numero, seme= folder.split(".")
    print(f"Total Progress:\t{ProgressBar.make_progress(card_num+1, len(card_folders))} {card_num+1}/{len(card_folders)}\nCurrent card: {numero} di {seme}")

    folder_path = os.path.join(INPUT_DIR_PATH, folder)
    card_output_dir_path= os.path.join(OUTPUT_DIR_PATH, folder)

    os.makedirs(card_output_dir_path, exist_ok=True)
    images= [img for img in os.listdir(folder_path) if img.endswith(".jpg")]

    for num, image in enumerate(images):
      if num + 1 > img_num_for_card:
        break

      print(f"\r\tCount: {ProgressBar.make_progress(num+1, img_num_for_card)} {num+1}/{img_num_for_card}", end="")

      # Viene caricata la carta
      card_path = os.path.join(folder_path, image)
      card_rgba = cv2.imread(card_path, cv2.IMREAD_UNCHANGED) # Carico anche alpha channel per la trasparenza
      if card_rgba is None:
        print(f"\n⚠️ Immagine non caricata correttamente: {card_path}, salto...")
        continue

      # Viene ridimensionata la carta a 200x200 (anche se in teoria sono già tutte di questa dimensione)
      card_rgba = cv2.resize(card_rgba, (200, 200))

      # Vengono separati i canali
      card_rgb = card_rgba[:,:,:3] # primi 3 canali

      # Ritaglio la parte di immagine che interessa a me, ovvero solo l'angolo in alto a sinistra contentente simbolo e numero
      card_cropped= card_rgb[0:80, 0:60]

      # Vengono applicate modifiche di luminosità casuali
      brightness = random.uniform(0.8, 1.2)
      card_cropped = np.clip(card_cropped * brightness, 0, 255).astype(np.uint8)


      # Viene salvata l'immagine
      img_name = f"{num + 1}.jpg"
      card_output_full_path = os.path.join(card_output_dir_path, img_name)
      cv2.imwrite(card_output_full_path, card_cropped)

      # Aggiorno il csv
      csv_writer.writerow([card_output_full_path, seme, numero])

    # Va su di tre righe e cancella il contenuto
    if card_num+1 != len(card_folders): sys.stdout.write('\033[F\033[K\033[F\033[K')

print("\n✅ Dataset creato con OpenCV!")
