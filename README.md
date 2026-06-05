# 🃏 Blackjack Card Detector — AI Lab Project

Progetto universitario realizzato per l'esame di **AI Lab** presso la **Sapienza Università di Roma**.  
AI Lab è un corso pratico incentrato sull'applicazione di tecniche di machine learning e computer vision su problemi reali, con sviluppo di modelli end-to-end.

L'obiettivo di questo progetto è riconoscere automaticamente le carte da gioco presenti in un'immagine (o in un feed video) e suggerire la mossa ottimale da eseguire a blackjack, seguendo la **strategia fondamentale**.
Il progetto è stato sviluppato in gruppo, io mi sono occupato dello sviluppo del modello di CNN e dello script per generarne il dataset, mentre i colleghi di corso:
- [Giovanni ciarra](https://github.com/GiovanniCiarra): si è occupato del lato di strategia fondamentale e riconoscimento delle carte con YOLO
- [Emanuele Bruni](https://github.com/MarcoLinardi): correzione bug, interazione tra i due modelli e sezione grafica
- [Marco Linardi](https://github.com/Emanuele-Bruni): correzione bug, interazione tra i due modelli  e sezione grafica

---

## 🧠 Come funziona

Il sistema utilizza una **pipeline a due modelli** che lavorano in sequenza:

1. **YOLO** — rileva le carte nell'immagine e ne estrae i bounding box
2. **CNN a doppia testa** — classifica ogni carta rilevata predicendo in parallelo il **numero** (A, 2–10, J, Q, K) e il **seme** (♥ ♦ ♣ ♠)

Il sistema divide visivamente l'inquadratura in due metà: la **metà superiore** è assegnata al dealer, quella **inferiore** al giocatore, ogniuna delle due parti di immagine vengono passate separatamente alla pipeline.

Il modello YOLO individuerà ogni carta nell'immagine andando poi a passare la carta rilevata alla CNN.

Il modello CNN andrà a rilevare il valore della carta per poi successivamente usare quel valore per suggerire una mossa tramite la strategia fondamentale 

---

## 📁 Struttura della repo

```
AI_Lab_project/
│
├── CNN_Double_head/          # Architettura e pesi della CNN a doppia testa
├── YOLO_cards_detector/      # Modello YOLO fine-tuned per il rilevamento carte
│
├── cnn_dataset_maker/        # Script per generare dataset sintetici per la CNN
├── yolo_dataset_maker/       # Script per generare dataset sintetici per YOLO
│
├── cli_utilities/            # Utility varie da riga di comando
├── test_images_complete_net/ # Immagini di test per la pipeline completa
├── REPORT/                   # Report scritto del progetto
│
├── main.py                              # Entry point principale con UI Gradio e supporto webcam
├── blackjack_card_detector.py           # Script alternativo: esegue YOLO + CNN su un'immagine statica
├── application_fundamental_strategy.py  # Logica della strategia di base del blackjack
├── ISTRUZIONI.md                        # Istruzioni dettagliate per training e utilizzo
└── .gitignore
```

---

## 🖥️ Interfaccia — `main.py`

Il file `main.py` è il punto di ingresso principale dell'applicazione e offre quattro modalità di utilizzo, selezionabili da riga di comando:

| Flag | Descrizione |
|------|-------------|
| `-s <path>` | Analizza un'immagine statica dal percorso specificato |
| `-g` | Avvia l'interfaccia grafica **Gradio** nel browser |
| `-lc <device>` | Usa la webcam locale del computer (es. `-lc 0`) |
| `-ic <url>` | Si connette a una webcam remota tramite URL |

### Interfaccia Gradio (`-g`)

L'interfaccia web costruita con [Gradio](https://gradio.app/) permette di:
- Caricare un'immagine o scattare una foto direttamente dalla webcam
- Visualizzare separatamente le carte del dealer e del giocatore con i rispettivi bounding box
- Leggere le carte identificate nei campi di testo dedicati
- Ricevere la **mossa consigliata** (HIT / STAND / DOUBLE / SPLIT) in tempo reale

### Modalità webcam (`-lc` / `-ic`)

In modalità live la pipeline analizza il feed video in tempo quasi reale, saltando i frame identici al precedente per evitare inferenze ridondanti. Su ogni frame vengono disegnati bounding box colorati (giallo per il dealer, azzurro per il giocatore) e la mossa suggerita viene mostrata al centro dello schermo.

---

## 🚀 Avvio rapido

```bash
# Interfaccia Gradio (consigliata)
python main.py -g

# Webcam locale
python main.py -lc 0

# Immagine statica
python main.py -s test_images_complete_net/test5.png
```

Per le istruzioni complete su come generare i dataset e allenare i modelli, consulta **[ISTRUZIONI.md](./ISTRUZIONI.md)**.

---

## ♟️ Strategia di base — `application_fundamental_strategy.py`

Implementa le tabelle della strategia di base del blackjack per tre casistiche: hard totals, soft totals (mano con asso) e coppie. Data la mano del giocatore e la carta scoperta del banco, restituisce la mossa ottimale:

```python
from application_fundamental_strategy import suggerisci_mossa

suggerisci_mossa(["A", "6"], "5")   # → "DOUBLE"
suggerisci_mossa(["9", "9"], "7")   # → "STAND"
suggerisci_mossa(["5", "3"], "6")   # → "DOUBLE"
```

---

## 🛠️ Tecnologie usate

- [PyTorch](https://pytorch.org/) — training e inferenza della CNN
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) — rilevamento oggetti
- [Gradio](https://gradio.app/) — interfaccia grafica web
- [OpenCV](https://opencv.org/) — elaborazione immagini e visualizzazione
- Python 3.x
