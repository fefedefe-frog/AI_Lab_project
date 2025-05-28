import torch
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
import torchmetrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from cli_utilities.simple_progress_bar import ProgressBar
from loops import training_loop, testing_loop
from CardDataset import CardDataset

# TODO: testare con solo i weight balancer, poi eventualmente testare separando
#  gli shared layers in due sezioni separate, direttamente attaccate alla loro testa
class DualHeadCNN(nn.Module):
    def __init__(self, num_classi_seme: int, num_classi_numeri: int) -> None:
        super(DualHeadCNN, self).__init__()

        # Sezione della cnn che si occupa del feature extraction
        self.featuresExtractor = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
        )

        # Sezione della cnn contenente le activation functions
        self.flatten = nn.Flatten()
        self.shared_layers = nn.Sequential(
            nn.Linear(256 * 30 * 22, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),
        )

        # Le due teste distinte per riconoscere i due tipi di classi differenti
        self.testa_seme = nn.Linear(256, num_classi_seme)
        self.testa_numero = nn.Linear(256, num_classi_numeri)

        # TODO: prima testare con solo i weights in loops.py, poi provare decommetando queste linee
        #   e commentando da self.shared_layer ... a self.testa_numero

        # self.testa_seme = nn.Sequential(
        #     nn.Linear(256 * 30 * 22, 256),
        #     nn.ReLU(),
        #
        #     nn.Linear(256, 256),
        #     nn.ReLU(),
        #
        #     nn.Linear(256, 256),
        #     nn.ReLU(),
        # )
        #
        # self.testa_numero = nn.Sequential(
        #     nn.Linear(256 * 30 * 22, 256),
        #     nn.ReLU(),
        #
        #     nn.Linear(256, 128),
        #     nn.ReLU(),
        #
        #     nn.Linear(128, 256),
        #     nn.ReLU(),
        #
        #     nn.Linear(256, 256),
        #     nn.ReLU(),
        # )


    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        x_features = self.featuresExtractor(x)      # Estrazione delle features
        x_flatten = self.flatten(x_features)        # Trasformazione da array 3D a 2D
        x_shared = self.shared_layers(x_flatten)    # Calcolo delle activation functions
        # TODO: eventualmente commentare la riga precendente a questa e passare direttamente x_flatten ai due logits
        
        logits_seme = self.testa_seme(x_shared)     # Predizione del seme
        logits_numero = self.testa_numero(x_shared) # Predizione del numero

        return logits_seme, logits_numero


epochs= 50
ProgressBar= ProgressBar()
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((240, 180)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    ])

    dataset = CardDataset("./cnn_dataset_maker/output", "./cnn_dataset_maker/output/dataset.csv", transform=transform)


    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualHeadCNN(num_classi_seme=len(dataset.classi_seme), num_classi_numeri=len(dataset.classi_numero)).to(device)


    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0004)

    metric_seme = torchmetrics.Accuracy(task='multiclass', num_classes=len(dataset.classi_seme)).to(device)
    metric_numero = torchmetrics.Accuracy(task='multiclass', num_classes=len(dataset.classi_numero)).to(device)

    # Dict di Array contenenti tutte le loss e le accuracy per poter fare il plot finale
    losses_seme: dict= {"train": [], "test": []}
    losses_numero: dict= {"train": [], "test": []}
    accuracy_seme: dict= {"train": [], "test": []}
    accuracy_numero: dict= {"train": [], "test": []}

    # Dict di Array per le Confusion Matrix
    all_labels: dict = {'seme': [], 'numero': []}
    all_preds: dict = {'seme': [], 'numero': []}

    for epoch in range(epochs):
        print(f"Epoch {ProgressBar.make_progress(epoch+1, epochs)} {epoch+1}/50")
        losses_seme['train'], losses_numero['train'], accuracy_seme['train'], accuracy_numero['train']= training_loop(model, train_loader, metric_seme, metric_numero, loss_fn, optimizer, device)
        all_labels, all_preds, losses_seme['test'], losses_numero['test'], accuracy_seme['test'], accuracy_numero['test']= testing_loop(model, train_loader, metric_seme, metric_numero, loss_fn, device)
    print("fatto!")

    torch.save(model, "cnn_allenata.pth")



    # Confusion Matrix per SEME
    cm_seme = confusion_matrix(all_labels['seme'], all_preds['seme'])
    disp_seme = ConfusionMatrixDisplay(confusion_matrix=cm_seme, display_labels=dataset.classi_seme)
    fig_seme, ax = plt.subplots()
    fig_seme.savefig("confusion_matrix_seme.png")
    disp_seme.plot(ax=ax)
    plt.title("Confusion Matrix - Seme")
    plt.show()

    # Confusion Matrix per NUMERO
    cm_numero = confusion_matrix(all_labels['numero'], all_preds['numero'])
    disp_numero = ConfusionMatrixDisplay(confusion_matrix=cm_numero, display_labels=dataset.classi_numero)
    fig_numero, ax = plt.subplots()
    fig_numero.savefig("confusion_matrix_numero.png")
    disp_numero.plot(ax=ax)
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
    plt.savefig("results.png")
    plt.close()
