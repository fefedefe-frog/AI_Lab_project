'''
Per la versione con reinforcement learning abbiamo scelto di usare la liberia Gymnasium, questa libreria
offre vari ambienti di gioco simulati, dove l'algoritmo di ML impara a giocare tramite un sistema di ricompense.
'''

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

# ----- Hyperparametri -----
GAMMA = 0.99
LR = 0.001
BATCH_SIZE = 64
MEMORY_SIZE = 10000
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995
EPISODES = 10000

# ----- Ambiente -----
env = gym.make("Blackjack-v1", sab=True)
n_actions = env.action_space.n


# ----- Preprocessing stato -----
def preprocess(state):
    # Stato = (player_sum, dealer_card, usable_ace) → normalizzazione
    return np.array([
        state[0] / 32.0,  # somma del giocatore (max 31 con asso multiplo)
        state[1] / 11.0,  # carta del dealer (1–10)
        int(state[2])  # usable_ace: True = 1, False = 0
    ], dtype=np.float32)


# ----- Rete neurale -----
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()

        self.input_layer = nn.Linear(3, 128)
        self.activation_fn = nn.ReLU()
        self.intermediate_layer = nn.Linear(128, 128)
        self.output_layer = nn.Linear(128, n_actions)

    def forward(self, x):
        output_from_input_layer= self.input_layer(x)
        output_from_activation_layer = self.activation_fn(output_from_input_layer)
        output_from_intermediate_layer = self.intermediate_layer(output_from_activation_layer)
        output_from_activation_layer_2 = self.activation_fn(output_from_intermediate_layer)

        final_output= self.output_layer(output_from_activation_layer_2)
        return final_output


# ----- Replay buffer -----
memory = deque(maxlen=MEMORY_SIZE)

# ----- Agent setup -----
policy_net = DQN()
target_net = DQN()
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=LR)
loss_fn = nn.MSELoss()

epsilon = EPSILON_START


# ----- Funzione di training -----
def train():
    if len(memory) < BATCH_SIZE:
        return

    batch = random.sample(memory, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(states)
    actions = torch.tensor(actions)
    rewards = torch.tensor(rewards)
    next_states = torch.tensor(next_states)
    dones = torch.tensor(dones, dtype=torch.bool)

    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
    next_q_values = target_net(next_states).max(1)[0]
    target = rewards + GAMMA * next_q_values * (~dones)

    loss = loss_fn(q_values, target.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# ----- Main loop -----
for episode in range(EPISODES):
    state, _ = env.reset()
    state = preprocess(state)
    done = False
    total_reward = 0

    while not done:
        if random.random() < epsilon:
            action = random.randrange(n_actions)
        else:
            with torch.no_grad():
                action = torch.argmax(policy_net(torch.tensor(state))).item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state_proc = preprocess(next_state)

        memory.append((state, action, reward, next_state_proc, done))
        state = next_state_proc
        total_reward += reward

        train()

    # Aggiorna la rete target periodicamente
    if episode % 20 == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # Decay epsilon
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    if episode % 500 == 0:
        print(f"Episode {episode}, Reward: {total_reward}, Epsilon: {epsilon:.2f}")