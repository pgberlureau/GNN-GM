import copy
from torch_geometric.data import Data
from torch_geometric import nn
from torch.nn.functional import one_hot
from torch_geometric.data import Dataset
from torch_geometric.transforms import BaseTransform
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import pandas as pd
import chess
import chess.pgn as PGN
import io
from random import shuffle
import json
import os
import os.path as osp
import fileinput
from math import tanh

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using gpu: " + str(torch.cuda.is_available()))


def line_to_graph(line):
    board = chess.Board(line["fen"])
    x = torch.zeros(64, 26)

    for k, v in board.piece_map().items():
        x[k, :7] = v.piece_type * one_hot(torch.tensor(v.piece_type), 7)
        x[k, 7:9] = one_hot(torch.tensor(v.color).long(), 2)

    x[:, 9:11] = one_hot(torch.tensor(board.turn).long(), 2)
    x[:, 11:13] = one_hot(torch.tensor(board.has_castling_rights(0)).long(), 2)
    x[:, 13:15] = one_hot(torch.tensor(board.has_castling_rights(0)).long(), 2)
    x[:, 15:17] = one_hot(torch.tensor(board.has_castling_rights(1)).long(), 2)
    x[:, 17:19] = one_hot(torch.tensor(board.has_castling_rights(1)).long(), 2)
    # x[:, 19:21] = one_hot(torch.tensor(board.is_repetition(2)).long(), 2)
    # x[:, 21:23] = one_hot(torch.tensor(board.is_repetition(3)).long(), 2)
    # x[:, 23] = board.fullmove_number
    # x[:, 24] = board.halfmove_clock
    x[:, 25] = -1 if board.ep_square is None else board.ep_square % 8

    edge_list = torch.tensor(list(
        map(lambda move: [move.from_square, move.to_square], board.legal_moves))).long()

    # print(line['evals'])
    cp = line['evals']['cp']

    """
    if -200 <= cp <= -50:
        y = torch.tensor(0)
    elif -50 < cp <= -25:
        y = torch.tensor(1)
    elif -25 < cp <= -10:
        y = torch.tensor(2)
    elif -10 < cp <= -5:
        y = torch.tensor(3)
    elif -5 < cp <= 0:
        y = torch.tensor(4)
    elif 0 < cp <= 5:
        y = torch.tensor(5)
    elif 5 < cp <= 10:
        y = torch.tensor(6)
    elif 10 < cp <= 25:
        y = torch.tensor(7)
    elif 25 < cp <= 50:
        y = torch.tensor(8)
    elif 50 < cp <= 200:
        y = torch.tensor(9)
    else:
        print("Error, cp is: "+str(cp))
        assert False
    """

    y = torch.tensor(cp >= 0).long()

    y = one_hot(y, num_classes=2).float()
    return Data(x=x, edge_index=edge_list, y=y)


# 10 epoch | 100_000 20min
# 
# 
# 90/10 Train/Test
#  1 Job avec 1_000_000

# RAM 20Go - 32Go
# Disk: 20Go 

# Conda

class Model(torch.nn.Module):
    def __init__(self, heads_nb=1, num_nodes=64, in_channels=26, hidden_channels=64, out_channels=10):
        super().__init__()
        self.heads_nb = heads_nb
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        self.lin0 = nn.Linear(in_channels, hidden_channels)
        self.gat = nn.GATv2Conv(
            in_channels=hidden_channels, out_channels=hidden_channels, heads=heads_nb)
        self.lin1 = nn.Linear(num_nodes * hidden_channels *
                              heads_nb, num_nodes * hidden_channels)
        self.lin2 = nn.Linear(num_nodes * hidden_channels, hidden_channels)
        self.lin3 = nn.Linear(hidden_channels, out_channels)

    def forward(self, g):
        y = self.lin0(g.x)
        y = y.relu()

        y = self.gat(x=y, edge_index=g.edge_index.T, edge_attr=g.edge_attr)
        y = y.reshape(-1)

        y = self.lin1(y)
        y = y.relu()
        y = y.reshape(-1)

        y = self.lin2(y)
        y = y.relu()
        y = y.reshape(-1)

        y = self.lin3(y)
        y = y.reshape(-1)

        return y


def load(line):
    kept = json.loads(line)
    to_save = {'fen': kept['fen'], 'evals': kept['evals'][0]['pvs'][0]}
    if not 'cp' in to_save['evals']:  # means the position is evaluated as a mate
        if to_save['evals']['mate'] > 0:
            to_save['evals']['cp'] = 20_000
        else:
            to_save['evals']['cp'] = -20_000

    to_save['evals']['cp'] = to_save['evals']['cp'] / 100
    return to_save


heads_nb = 16
model = Model(heads_nb=heads_nb, out_channels=2)

print("Model created")

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

model.to(device)
model.train()

data_size = 1e7
batch_size = 1024
num_epochs = 2
losses = []
accs = []

for epoch in range(num_epochs):
    print("Epoch: " + str(epoch))

    epoch_loss = 0.
    epoch_acc = 0.

    num_ones = 0
    num_zeros = 0

    with open('lichess_db_eval.jsonl') as f:
        batch_loss = 0.
        batch_acc = 0.

        for i, line in enumerate(f):
            line = load(line)
            g = line_to_graph(line).to(device)

            output = model(g)

            loss = criterion(output, g.y)

            if torch.argmax(g.y):
                num_ones += 1
            else:
                num_zeros += 1

            batch_loss += loss / batch_size
            batch_acc += (torch.argmax(output) == torch.argmax(g.y)
                          ).detach().item() / batch_size

            epoch_loss += loss.detach().item() / data_size
            epoch_acc += (torch.argmax(output) == torch.argmax(g.y)
                          ).detach().item() / data_size

            if i % batch_size == 0 and i != 0:
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                batch_loss = batch_loss.detach().item()

                print("Currently at i=" + str(i))
                print("\t Batch loss is: " + str(batch_loss))
                print("\t Batch acc is: " + str(batch_acc))

                batch_loss = 0.
                batch_acc = 0.

            if i > data_size:
                break

    print("Epoch loss is: " + str(epoch_loss))
    print("Epoch acc is: " + str(epoch_acc))
    print("Num zeros: " + str(num_zeros))
    print("Num ones: " + str(num_ones))

    losses.append(epoch_loss)
    accs.append(epoch_acc)

    torch.save(model.state_dict(), 'model.pth')

fig, ax = plt.subplots(2)
ax[0].plot(losses)
ax[0].set_title('Losses')
ax[1].plot(accs)
ax[1].set_title('Accuracies')
plt.savefig('losses.png')
