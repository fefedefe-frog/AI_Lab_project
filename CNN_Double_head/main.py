import argparse
import sys
import os
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
import torchmetrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from cli_utilities.simple_progress_bar import ProgressBar
from CNN_Double_head.loops import training_loop, testing_loop
from CNN_Double_head.CardDataset import CardDataset
from CNN_Double_head.DualHeadCNN import DualHeadCNN


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
SCRIPT_DIR= os.path.dirname(os.path.realpath(__file__))
OUTPUT_RESULT_FOLDER= "output"
DEFAULT_EPOCHS= 10
DEFAULT_BATCH_SIZE= 32


def run_train(dataset_path: str, csv_path: str, batch_size: int, epochs: int, result_path: str) -> None:
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((240, 180)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    ])

    dataset = CardDataset(dataset_path, csv_path, transform=transform)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualHeadCNN(num_classi_seme=len(dataset.classi_seme), num_classi_numeri=len(dataset.classi_numero)).to(
        device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0004)

    metric_seme = torchmetrics.Accuracy(task='multiclass', num_classes=len(dataset.classi_seme)).to(device)
    metric_numero = torchmetrics.Accuracy(task='multiclass', num_classes=len(dataset.classi_numero)).to(device)

    # Dict di Array contenenti tutte le loss e le accuracy per poter fare il plot finale
    losses_seme: dict = {"train": [], "test": []}
    losses_numero: dict = {"train": [], "test": []}
    accuracy_seme: dict = {"train": [], "test": []}
    accuracy_numero: dict = {"train": [], "test": []}

    # Dict di Array per le Confusion Matrix
    all_labels: dict = {'seme': [], 'numero': []}
    all_preds: dict = {'seme': [], 'numero': []}

    for epoch in range(epochs):
        print(f"\t\t== Epoch {epoch + 1} ==")
        (
            train_loss_seme,
            train_loss_numero,
            train_accuracy_seme,
            train_accuracy_numero
        ) = training_loop(model, train_loader, metric_seme, metric_numero, loss_fn, optimizer, device)

        losses_seme['train'].extend(train_loss_seme)
        losses_numero['train'].extend(train_loss_numero)
        accuracy_seme['train'].extend(train_accuracy_seme)
        accuracy_numero['train'].extend(train_accuracy_numero)

        (all_labels,
         all_preds,
         test_loss_seme,
         test_loss_numero,
         test_accuracy_seme,
         test_accuracy_numero
         ) = testing_loop(model, test_loader, metric_seme, metric_numero, loss_fn, device)

        losses_seme['test'].extend(test_loss_seme)
        losses_numero['test'].extend(test_loss_numero)
        accuracy_seme['test'].extend(test_accuracy_seme)
        accuracy_numero['test'].extend(test_accuracy_numero)

        # Ritorno su di 12 righe (usate dai print nel training_loop) e 4 (usate nel testing_loop) nel terminale così da avere sempre e solo le 13 righe che si aggiornano ogni ciclo
        if epoch + 1 != epochs: sys.stdout.write("\033[F" * 16)
    print("fatto!")

    # Test per vedere cosa sta dentro sti array
    #print("acc_seme_train: ", len(accuracy_seme["train"]), accuracy_seme['train'])
    #print("acc_seme_test: ", len(accuracy_seme["test"]), accuracy_seme['test'])

    torch.save(model.state_dict(), os.path.join(result_path, "cnn_weights.pth"))

    # Confusion Matrix per SEME
    cm_seme = confusion_matrix(np.concatenate(all_labels['seme']), np.concatenate(all_preds['seme']))
    disp_seme = ConfusionMatrixDisplay(confusion_matrix=cm_seme, display_labels=dataset.classi_seme)
    fig_seme, ax = plt.subplots()
    disp_seme.plot(ax=ax)
    fig_seme.savefig(os.path.join(result_path, "confusion_matrix_seme.png"))
    plt.title("Confusion Matrix - Seme")
    plt.show()

    # Confusion Matrix per NUMERO
    cm_numero = confusion_matrix(np.concatenate(all_labels['numero']), np.concatenate(all_preds['numero']))
    disp_numero = ConfusionMatrixDisplay(confusion_matrix=cm_numero, display_labels=dataset.classi_numero)
    fig_numero, ax = plt.subplots()
    disp_numero.plot(ax=ax)
    fig_numero.savefig(os.path.join(result_path, "confusion_matrix_numero.png"))
    plt.title("Confusion Matrix - Numero")
    plt.show()

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    axs[0, 0].plot(losses_seme['train'], label="Train Loss Seme", color='blue')
    axs[0, 0].plot(losses_seme['test'], label="Test Loss Seme", color='orange')
    axs[0, 0].set_title("Loss Seme")
    axs[0, 0].legend()

    axs[0, 1].plot(losses_numero['train'], label="Train Loss Numero", color='green')
    axs[0, 1].plot(losses_numero['test'], label="Test Loss Numero", color='red')
    axs[0, 1].set_title("Loss Numero")
    axs[0, 1].legend()

    axs[1, 0].plot(accuracy_seme['train'], label="Train Accuracy Seme", color='purple')
    axs[1, 0].plot(accuracy_seme['test'], label="Test Accuracy Seme", color='pink')
    axs[1, 0].set_title("Accuracy Seme")
    axs[1, 0].legend()

    axs[1, 1].plot(accuracy_numero['train'], label="Accuracy Numero", color='cyan')
    axs[1, 1].plot(accuracy_numero['test'], label="Test Accuracy Numero", color='brown')
    axs[1, 1].set_title("Accuracy Numero")
    axs[1, 1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(result_path, "results.png"))
    plt.close()


parser= argparse.ArgumentParser(description="CNN trainer script, this script launch the train for the cnn")
parser.add_argument("dataset_path", type=str, help="path to the dataset folder")
parser.add_argument("csv_path", type=str, help="path to the csv file")
parser.add_argument("epochs", type=int, help="how many epochs the model will train")
parser.add_argument("-b", "--batch_size", type=int, help="the size of the batch")
parser.add_argument("-o", "--output_dir", type=str, help="path where will be saved the result")

ProgressBar= ProgressBar()
if __name__ == "__main__":
    args= parser.parse_args()

    # Recupero datasaet e csv path
    dt_path= f"{Path(args.dataset_path).resolve()}"
    c_path= f"{Path(args.csv_path).resolve()}"

    # Recupero numero delle epochs
    e= args.epochs

    # Se definito recupero il numero del batch size,
    # sennò assume il valore di default 32
    b_size= DEFAULT_BATCH_SIZE
    if args.batch_size is not None:
        b_size= args.batch_size

    # Se definito recupero il path dove salvare i risultati,
    # sennò viene salvato in automatico nella cartella dello script
    output_path= f"{SCRIPT_DIR}/result"
    if args.output_dir is not None:
        output_path= f"{Path(args.output_dir).resolve()}/result"

    # Controllo se il nome della cartella dei risultati è disponibile,
    # sennò aggiungo un numero e controllo di nuovo
    i= 0
    _folder_name= f"{OUTPUT_RESULT_FOLDER}_e{e}_bs{b_size}"
    while os.path.exists(os.path.join(output_path, _folder_name)):
        i += 1
        _folder_name = f"{OUTPUT_RESULT_FOLDER}_e{e}_bs{b_size}_{i}"
    output_path= os.path.join(output_path, _folder_name)
    os.makedirs(output_path, exist_ok=True)

    print(f"Input path:\n  -dataset:\t{dt_path}\n  -csv:\t\t{c_path}")
    print(f"epochs:\t{e:>5}")
    print(f"batch size:\t{b_size}")
    print(f"Output path:\n  -saved in:\t{output_path}\n  -folder name:\t{_folder_name}\n")

    run_train(dt_path, c_path, b_size, e, output_path)