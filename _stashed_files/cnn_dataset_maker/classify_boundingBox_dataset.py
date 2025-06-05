import os
import csv
import cv2

# === CONFIGURAZIONE ===
SCRIPT_DIR= os.path.dirname(os.path.realpath(__file__))
CARTELLA_IMMAGINI = f"{SCRIPT_DIR}/boundingBox_dataset"
CSV_PATH = f"{SCRIPT_DIR}/output/dataset.csv"
PATH_STATICO = "boundingBox_dataset"  # usato come valore "path" nel CSV

# === CONTROLLA SE CSV ESISTE E APRILO IN MODALITÀ APPEND ===
file_esiste = os.path.exists(CSV_PATH)

with open(CSV_PATH, mode='a', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # Scrivi header solo se il file non esiste
    if not file_esiste:
        writer.writerow(['path', 'seme', 'numero'])

    # Ordina i file per nome
    immagini = sorted(os.listdir(CARTELLA_IMMAGINI))

    for nome_file in immagini:
        if not nome_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        path_img = os.path.join(CARTELLA_IMMAGINI, nome_file)
        img = cv2.imread(path_img)

        if img is None:
            print(f"[!] Errore nel leggere: {nome_file}")
            continue

        # Mostra immagine
        cv2.imshow("Immagine da etichettare", img)
        print(f"\n✏️  Etichettatura per: {nome_file}")
        cv2.waitKey(1)  # Serve per aggiornare correttamente la finestra OpenCV

        # Chiedi seme e numero
        seme = input("→ Inserisci seme (H, D, C, S): ").strip().upper()
        numero = input("→ Inserisci numero (A,2,...,10,J,Q,K): ").strip().upper()

        # Chiudi finestra immagine
        cv2.destroyAllWindows()

        # Scrivi nel CSV
        writer.writerow([f"{PATH_STATICO}/{nome_file}",seme,numero])
        print(f"✅ Salvato: {nome_file} → {seme} {numero}")
