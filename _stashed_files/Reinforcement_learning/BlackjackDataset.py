'''
per il supervised learning tramite dataset dobbiamo creare un dataset di mosse ottimali
da poter passare all'algoritmo di ML, per fare ciò usiamo gymnasium
'''
import gymnasium as gym
import csv
import pandas as pd
from torch.utils.data import Dataset


class BlackjackDataset(Dataset):
    def __init__(self, X: []= None, y: []= None):
        self.X = X if X is not None else []
        self.y = y if y is not None else []

        self.ds_path= ""
        self.df= pd.DataFrame()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]

    def set_dataset_path(self, path: str) -> None:
        self.ds_path= path

    def get_dataset_path(self) -> str:
        return self.ds_path

    def get_dataframe(self) -> pd.DataFrame:
        return self.df

    def make_dataset(self, strategy_type: str = "base", simulated_hands: int = 100_000, risk: float = 0.4) -> pd.DataFrame | None:
        env = gym.make("Blackjack-v1", sab=True)

        # mi assicuro che il valore del rischio sia compreso tra 0-1
        risk = min(max(0.0, abs(risk)), 1.0)
        if strategy_type not in ["base", "greedy", "probabilistic"]:
            return None

        with open(f"blackjack_dataset_{strategy_type}.csv", mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['player_sum', 'dealer_card', 'usable_ace', 'action'])

            for _ in range(simulated_hands):  # numero di mani da simulare
                state, _info = env.reset()
                player_sum, dealer_card, usable_ace = state

                action = ""
                if strategy_type == "base":
                    action = self.base_strategy(player_sum, dealer_card, usable_ace)
                elif strategy_type == "greedy":
                    action = self.greedy_strategy(player_sum)
                elif strategy_type == "probabilistic":
                    action = self.probabilistic_strategy(player_sum, risk)

                # Esegui l'azione nel passo successivo
                state, reward, done, truncated, info = env.step(action)

                writer.writerow([player_sum, dealer_card, int(usable_ace), action])

        self.df= pd.read_csv(f"blackjack_dataset_{strategy_type}.csv")
        return self.df

    # strategia base che mimizza le perdite su lungo termine
    # player_sum: somma delle carte del giocatore (1,...21)
    # dealer_card: valore della carta visibile del dealer (2-11)
    # usable_ace: indica se l'asso del giocatore può valere 11 senza sballare
    def base_strategy(self, player_sum: int, dealer_card: int, usable_ace: bool) -> int:
        if player_sum >= 20:
            return 0
        elif player_sum <= 11:
            return 1
        elif usable_ace and player_sum >= 18:
            return 0
        elif 2 <= dealer_card <= 6:
            if player_sum >= 12:
                return 0
            else:
                return 1
        else:
            return 1

    # strategia  che porta il giocatore a pescare se ha meno di 17 come somma delle sue carte
    def greedy_strategy(self, player_sum: int) -> int:
        return 1 if player_sum < 17 else 0

    # strategia basata sulla probabilità di sballare in base alla somma di carte che si ha
    def probabilistic_strategy(self, player_sum: int, risk_level: float= 0.4) -> int:
        bust_cards= max(0, player_sum - 21)
        prob_bus= bust_cards / 13
        return 1 if prob_bus < risk_level else 0

