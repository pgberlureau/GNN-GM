from torch_geometric import nn
from torch_geometric.loader import DataLoader
import torch

from dataset import GraphChessDataset


# 10 epoch | 100_000 20min
#
#
# 90/10 Train/Test
#  1 Job avec 1_000_000

# RAM 20Go - 32Go
# Disk: 20Go

# Conda

class Model(torch.nn.Module):
    def __init__(self, heads_nb=1, num_nodes=64, in_channels=15, hidden_channels=64, out_channels=10):
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

        print(y.shape)

        y = self.gat(x=y, edge_index=g.edge_index, edge_attr=g.edge_attr)

        print(y.shape)

        y = self.lin1(y)
        y = y.relu()
        y = y.reshape(-1)

        print(y.shape)

        y = self.lin2(y)
        y = y.relu()
        y = y.reshape(-1)

        print(y.shape)

        y = self.lin3(y)
        y = y.reshape(-1)

        return y


def train(root, num_epochs, batch_size):
    train_dataset = GraphChessDataset(root, cp_transform=None)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = Model(heads_nb=16, out_channels=2)
    print("Model created")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            print(data)

            output = model(data)

            loss = criterion(output, data.y)
            loss.backward()

            optimizer.step()

        # losses.append(epoch_loss)
        # accs.append(epoch_acc)

    torch.save(model.state_dict(), 'model.pth')

    # fig, ax = plt.subplots(2)
    # ax[0].plot(losses)
    # ax[0].set_title('Losses')
    # ax[1].plot(accs)
    # ax[1].set_title('Accuracies')
    # plt.savefig('losses.png')


if __name__ == '__main__':
    train("/media/gabriel/Chess/out", 20, 4)
