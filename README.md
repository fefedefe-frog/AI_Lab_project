# Blackjack Card Detector — AI Lab Project

Progetto universitario realizzato per l'esame di **AI Lab** presso la **Sapienza Università di Roma**.  
AI Lab è un corso pratico incentrato sull'applicazione di tecniche di machine learning e computer vision su problemi reali, con sviluppo di modelli end-to-end.

L'obiettivo è riconoscere automaticamente le carte da gioco presenti in un'immagine o in un feed video in tempo reale e suggerire la mossa ottimale da eseguire a blackjack, seguendo la **strategia fondamentale**.

Il progetto è stato sviluppato in gruppo:
- [Marco Linardi](https://github.com/MarcoLinardi): correzione bug, interazione tra i due modelli e sezione grafica
- [Emanuele Bruni](https://github.com/Emanuele-Bruni): correzione bug, interazione tra i due modelli e sezione grafica
- [Giovanni Ciarra](https://github.com/GiovanniCiarra): strategia fondamentale e rilevamento carte con YOLO
- Sviluppatore CNN: architettura della rete neurale e generazione del dataset sintetico

---

## Come funziona

Il sistema implementa una **pipeline a due modelli in cascata** che operano in sequenza su ogni frame o immagine in ingresso.

### Rilevamento — YOLO

Il primo stadio della pipeline utilizza un modello **YOLOv8s** fine-tuned su un dataset sintetico di carte da gioco. Il modello individua tutte le carte presenti nell'immagine e ne restituisce i bounding box con la relativa confidenza. Il dataset di training è stato generato sinteticamente: carte da gioco vengono posizionate su background casuali con variazioni di scala, rotazione e sovrapposizione parziale, replicando le condizioni di ripresa reali.

### Classificazione — CNN a doppia testa

Per ogni carta rilevata da YOLO, la regione di interesse viene ritagliata, ridimensionata a 240×180 pixel e passata a una **CNN con architettura dual-head**. La rete condivide un unico feature extractor (quattro strati convoluzionali) e poi si divide in due teste di classificazione indipendenti:

- **Testa numero**: predice il valore della carta in 13 classi (A, 2–10, J, Q, K)
- **Testa seme**: predice il seme in 4 classi (cuori, quadri, fiori, picche)

Questa architettura permette di apprendere una rappresentazione visiva comune della carta e specializzarsi separatamente sulle due proprietà da classificare. Anche il dataset per la CNN è sintetico, generato a partire da template base delle carte con augmentation.

### Divisione spaziale dealer / player

Il sistema adotta una convenzione spaziale: la **metà superiore** dell'inquadratura è assegnata al dealer, la **metà inferiore** al giocatore. Ogni bounding box viene assegnato all'uno o all'altro in base alla coordinata verticale del suo centro. Questa divisione funziona bene nel contesto d'uso previsto — un tavolo ripreso dall'alto — ed elimina la necessità di un modello dedicato al riconoscimento del ruolo.

### Strategia fondamentale

Una volta classificate tutte le carte, il sistema calcola il totale della mano del giocatore (gestendo correttamente gli assi come 1 o 11) e consulta le tabelle della **basic strategy** del blackjack, implementate per tre scenari: hard totals, soft totals e coppie. Il risultato è la mossa ottimale statistica: **HIT**, **STAND**, **DOUBLE** o **SPLIT**.

---

## Architettura della repo

```
AI_Lab_project/
│
├── CNN_Double_head/          # Definizione della rete DualHeadCNN, training loop e pesi
├── YOLO_cards_detector/      # Script di training e pesi del modello YOLO fine-tuned
│
├── cnn_dataset_maker/        # Generazione del dataset sintetico per la CNN
├── yolo_dataset_maker/       # Generazione del dataset sintetico per YOLO
│
├── cli_utilities/            # Utility da riga di comando condivise
├── test_images_complete_net/ # Immagini di test per la pipeline completa
├── REPORT/                   # Report scritto del progetto
│
├── main.py                              # Entry point: Gradio, webcam locale/remota, immagine statica
├── blackjack_card_detector.py           # Inference su immagine statica (pipeline completa)
├── application_fundamental_strategy.py  # Tabelle e logica della strategia fondamentale
└── ISTRUZIONI.md                        # Istruzioni per la generazione dei dataset e il training
```

---

## Tecnologie

| Libreria | Utilizzo |
|----------|----------|
| [PyTorch](https://pytorch.org/) | Architettura e training della CNN dual-head |
| [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) | Rilevamento carte (object detection) |
| [Gradio](https://gradio.app/) | Interfaccia grafica web interattiva |
| [OpenCV](https://opencv.org/) | Elaborazione immagini, visualizzazione bounding box, gestione webcam |
| [scikit-learn](https://scikit-learn.org/) | Confusion matrix e metriche di valutazione |
| Python 3.x | Linguaggio principale |
