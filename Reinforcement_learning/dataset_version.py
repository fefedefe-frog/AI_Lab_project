import torch
import torchmetrics
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset, random_split
from torch import nn

from Progetto.BlackjackDataset import BlackjackDataset



# hyperparameters
epochs= 10   # usa il train per n volte
batch_size= 64
learning_rate= 0.0001


train_size= 0.8
test_size= 1 - train_size


bj_dataset= BlackjackDataset()
dataframe= None
if bj_dataset.get_dataset_path() == "":
    dataframe= bj_dataset.make_dataset(strategy_type="base", simulated_hands=500_000)
else:
    dataframe= bj_dataset.get_dataframe()

# Controllo di avere dei dati validi
print(dataframe.head())

# Separo gli input dai risultati
X= dataframe[['player_sum', 'dealer_card', 'usable_ace']].values
y= dataframe['action'].values

# Converto i dati in un formato valido per pytorch
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Converto il tutto in un dataset pytorch
torch_dataset= TensorDataset(X_tensor, y_tensor)

# Divido il dataset in train e test
training_data, test_data= random_split(torch_dataset, [train_size, test_size])

# creo il dataloader
# il dataloader è il programma che recupera i dati dal disco e li passa al modello
train_loader = DataLoader(training_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)

# creo il modello
device= ('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    print('Torch.cuda')
else: print("Torch.cpu")
class BlackjackModel(nn.Module):
    def __init__(self):
        # specificamio la struttura del modello
        super().__init__()

        # metodo alternativo, creiamo solo i layer che ci servono e li connetteremo nella forward function
        self.input_layer = nn.Linear(3, 64)
        self.activation_fn = nn.CELU()
        self.intermediate_layer= nn.Linear(64, 64)
        self.output_layer = nn.Linear(64, 2)

    def forward(self, x):

        # connettiamo i layer specificati nella __init__ function (per il metodo alternativo)
        x_output_input_layer= self.input_layer(x)
        x_output_activation_layer = self.activation_fn(x_output_input_layer)

        x_output_intermediate= self.intermediate_layer(x_output_activation_layer)
        x_output_activation_layer_2= self.activation_fn(x_output_intermediate)

        x_output_intermediate_2= self.intermediate_layer(x_output_activation_layer_2)
        x_output_activation_layer_3= self.activation_fn(x_output_intermediate_2)

        # metodo alternativo
        logits= self.output_layer(x_output_activation_layer_3)

        return logits

# inizializzo il modello
model= BlackjackModel().to(device)

# Definisco la funzione di perdita
loss_fn= nn.CrossEntropyLoss()
#loss_fn= nn.MSELoss()
#loss_fn= nn.BCEWithLogitsLoss()

# definisco l'ottimizzatore
optimizer= optim.SGD(model.parameters(), lr=learning_rate)

# Creiamo la matrice di accuratezza
metric= torchmetrics.Accuracy(task='multiclass', num_classes=2).to(device)

#metric= torchmetrics.classification.BinaryAccuracy()

def training_loop(dataloader, model, loss_fn, optimizer):

    dataset_size= len(dataloader)
    # recupero il batch di dati dal disco
    for batch, (X,y) in enumerate(dataloader):
        y= y.long()
        # calcolo le previsioni del modello
        pred= model(X)

        # calcolo l'errore tra le nostre previsioni e i label corretti
        loss= loss_fn(pred, y)

        # eseguo il passaggio di backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 500 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"Loss: {loss}, [{current:>5} / {dataset_size}]")
            acc= metric(pred, y)
            print(f"Accuracy: {acc} || {acc*100:.2f}%")

    # stmpo l'accuratezza alla fine del training
    acc= metric.compute()
    print(f"Final Training Accuracy: {acc}  || {acc*100:.2f}%")
    metric.reset()

def testing_loop(dataloader, model, loss_fn):
    # disabilito l'aggiornamento dei pesi
    with torch.no_grad():
        for X, y in dataloader:
            y= y.long()
            pred= model(X)
            loss= loss_fn(pred, y)
            acc= metric(pred, y)

    # stampo direttamente l'accuratezza finale
    acc= metric.compute()
    print(f"Final Testing Accuracy: {acc} || {acc*100:.2f}%")
    metric.reset()

# avvio l'allenamento del modello
for e in range(epochs):
    print(f"\n===== Epoch {e}")
    training_loop(train_loader, model, loss_fn, optimizer)
    testing_loop(test_loader, model, loss_fn)
print("Fatto!")
