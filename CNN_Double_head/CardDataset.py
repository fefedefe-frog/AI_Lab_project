import cv2
import pandas as pd

import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CardDataset(Dataset):
    def __init__(self, dataset_path, csv_file_path: str, transform: transforms) -> None:

        # Carico il csv usando pandas
        self.dataset_path= dataset_path
        self.csv_data = pd.read_csv(csv_file_path)
        self.transforms = transform

        # Definisco le classi per seme e numeri
        self.classi_seme: tuple= ("cuori", "quadri", "fiori", "picche")
        self.classi_numero: tuple= ('A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K')

        # Converto le etichette testuali in indici numerici usabili dalla rete neurale
        self.seme_to_idx = {cls: idx for idx, cls in enumerate(self.classi_seme)}
        self.numero_to_idx = {cls: idx for idx, cls in enumerate(self.classi_numero)}

    def __len__(self) -> int:
        return len(self.csv_data)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        riga = self.csv_data.iloc[idx]
        
        path_clean = str(riga['image_path']).replace('\\', '/')
        image_path = f"{self.dataset_path}/{path_clean}"
        seme = self.seme_to_idx[riga["seme"]]
        numero = self.numero_to_idx[riga["numero"]]

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"{image_path} not found")

        if self.transforms:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            out_img = self.transforms(image)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (240, 180))
            image = image.astype(np.float32) / 255.0
            image = np.transpose(image, (2, 0, 1))
            out_img = torch.tensor(image, dtype=torch.float32)

        return out_img, torch.tensor(seme, dtype=torch.long), torch.tensor(numero, dtype=torch.long)