import os

labels_dir = "yolov5/data/blackjack/test/labels"
new_labels_dir = "yolov5/data/blackjack/test/labels"
os.makedirs(new_labels_dir, exist_ok=True)

for file in os.listdir(labels_dir):
    if not file.endswith(".txt"):
        continue

    path = os.path.join(labels_dir, file)
    with open(path, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 5:
            parts[0] = '0'  # Sostituisce la classe con 0
            new_lines.append(' '.join(parts) + '\n')

    with open(os.path.join(new_labels_dir, file), 'w') as f:
        f.writelines(new_lines)

    print(f"✔️ Etichetta convertita: {file}")

print("\n✅ Conversione completata: tutte le etichette ora hanno classe '0' e sono salvate in /labels")
