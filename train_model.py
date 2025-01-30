import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
from torch import nn
from torch_geometric import nn as nn_geo
from torch_geometric.loader import DataLoader
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score

from dataset import build_train_test_dataset
from matplotlib import pyplot as plt


def plot_metrics(out_dir, num_epochs, model_type, train_loss_list, train_accuracy_list, train_precision_list,
                 train_recall_list,
                 test_loss_list, test_accuracy_list, test_precision_list, test_recall_list):
    # Training Metrics
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(range(num_epochs), train_loss_list, label='loss')
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_title(f"Training Loss\nModel: {model_type}")
    ax.legend()
    fig.savefig(out_dir / f"train_loss.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(num_epochs), train_precision_list, label='precision')
    ax.plot(range(num_epochs), train_recall_list, label='recall')
    ax.plot(range(num_epochs), train_accuracy_list, label='accuracy')
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Metrics")
    ax.set_title(f"Training Metrics\nModel: {model_type}")
    ax.legend()
    fig.savefig(out_dir / f"train.png")
    plt.close(fig)

    # Test Metrics
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(range(num_epochs), test_loss_list, label='test loss')
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_title(f"Test Loss\nModel: {model_type}")
    ax.legend()
    fig.savefig(out_dir / f"test_loss.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(num_epochs), test_precision_list, label='test precision')
    ax.plot(range(num_epochs), test_recall_list, label='test recall')
    ax.plot(range(num_epochs), test_accuracy_list, label='test accuracy')
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Metrics")
    ax.set_title(f"Test Metrics\nModel: {model_type}")
    ax.legend()
    fig.savefig(out_dir / f"test.png")
    plt.close(fig)


CHESS_EMBEDDING_SIZE = 15
CHESS_BOARD_SIZE = 64


class GNNModel(torch.nn.Module):
    def __init__(self, heads_nb=1, hidden_channels=64, out_channels=10):
        super().__init__()
        self.heads_nb = heads_nb
        self.in_channels = CHESS_EMBEDDING_SIZE
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        self.lin0 = nn.Linear(CHESS_EMBEDDING_SIZE, hidden_channels)
        self.gat = nn_geo.GATv2Conv(in_channels=hidden_channels, out_channels=hidden_channels, heads=heads_nb)
        self.lin1 = nn.Linear(CHESS_BOARD_SIZE * hidden_channels * heads_nb, CHESS_BOARD_SIZE * hidden_channels)
        self.lin2 = nn.Linear(CHESS_BOARD_SIZE * hidden_channels, hidden_channels)
        self.lin3 = nn.Linear(hidden_channels, out_channels)

    def forward(self, g):
        y = self.lin0(g.x)
        y = y.relu()

        y = self.gat(x=y, edge_index=g.edge_index, edge_attr=g.edge_attr)

        y = y.reshape((g.batch_size, CHESS_BOARD_SIZE * self.heads_nb * self.hidden_channels))

        y = self.lin1(y)
        y = y.relu()

        y = self.lin2(y)
        y = y.relu()

        y = self.lin3(y)

        return y


class MLPModel(torch.nn.Sequential):
    def __init__(self, out_channels=10):
        in_channels = CHESS_BOARD_SIZE * CHESS_EMBEDDING_SIZE
        hidden_channels = 2 * in_channels
        super().__init__(nn.Linear(in_channels, hidden_channels),
                         nn.ReLU(),
                         nn.Linear(hidden_channels, hidden_channels),
                         nn.ReLU(),
                         nn.Linear(hidden_channels, out_channels))

    def forward(self, g):
        return super().forward(torch.flatten(g.x, -2, -1))


def centipawn_transform(cp: torch.int16) -> torch.Tensor:
    res = torch.zeros(2, dtype=torch.float)
    res[int(cp >= 0)] = 1
    return res


def test(model, criterion, test_dataloader, device):
    total_test_loss = 0.0
    test_preds = []
    test_labels = []

    with torch.no_grad():
        for data in test_dataloader:
            data = data.to(device)
            output = model(data)
            loss = criterion(output, data.y)
            total_test_loss += loss.item()

            preds = torch.argmax(output, dim=1).cpu().numpy()
            labels = torch.argmax(data.y, dim=1).cpu().numpy()

            test_preds.extend(preds)
            test_labels.extend(labels)

    # Compute test metrics
    test_loss = total_test_loss / len(test_dataloader)
    test_accuracy = accuracy_score(test_labels, test_preds)
    test_precision = precision_score(test_labels, test_preds, average='weighted', zero_division=0)
    test_recall = recall_score(test_labels, test_preds, average='weighted', zero_division=0)

    return test_loss, test_accuracy, test_precision, test_recall


def train(root: str, dataset_size: int, num_epochs: int, batch_size: int, load_workers: int, out_dir: str, gnn: bool):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_type = "gnn" if gnn else "mlp"
    out_path = Path(out_dir) / f"{model_type}_{timestamp}"
    out_path.mkdir(exist_ok=False)

    # concat_x controls how the x field is concatenated by TorchGeometric.
    # Use False to create a new axis for the concatenation.

    # cp_transform controls how the centipawn data is transformed. If not present, it is a torch.int16 integer.
    train_dataset, test_dataset = build_train_test_dataset(root, dataset_size,
                                                           cp_transform=centipawn_transform, concat_x=gnn)

    # load_workers controls how many process are used to load the data in the background.
    # More process can be make the loading quicker if the CPU is the bottleneck

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=load_workers)
    test_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=load_workers)

    nb_batch = len(train_dataset) // batch_size

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if gnn:
        model = GNNModel(heads_nb=16, out_channels=2)
    else:
        model = MLPModel(out_channels=2)

    print("Model created")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.to(device)
    model.train()

    train_loss_list = []
    train_accuracy_list = []
    train_precision_list = []
    train_recall_list = []

    test_loss_list = []
    test_accuracy_list = []
    test_precision_list = []
    test_recall_list = []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        total_loss = 0.0
        all_preds = []
        all_labels = []

        for data in tqdm(train_loader, total=nb_batch):
            data = data.to(device)
            optimizer.zero_grad()

            # Forward pass
            output = model(data)

            # Compute loss
            loss = criterion(output, data.y)
            total_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Collect predictions and labels
            preds = torch.argmax(output, dim=1).cpu().numpy()
            labels = torch.argmax(data.y, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

        # Compute metrics
        epoch_loss = total_loss / len(train_loader)
        epoch_accuracy = accuracy_score(all_labels, all_preds)
        epoch_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        epoch_recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)

        test_loss, test_accuracy, test_precision, test_recall = test(model, criterion, test_loader, device)

        train_loss_list.append(epoch_loss)
        train_accuracy_list.append(epoch_accuracy)
        train_precision_list.append(epoch_precision)
        train_recall_list.append(epoch_recall)

        test_loss_list.append(test_loss)
        test_accuracy_list.append(test_accuracy)
        test_precision_list.append(test_precision)
        test_recall_list.append(test_recall)

        # Print epoch  statistics
        print(f"Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, "
              f"Precision: {epoch_precision:.4f}, Recall: {epoch_recall:.4f}")
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, "
              f"Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}")

    torch.save(model.state_dict(), out_path / "model.pth")

    plot_metrics(out_path, num_epochs, model_type, train_loss_list, train_accuracy_list, train_precision_list,
                 train_recall_list,
                 test_loss_list, test_accuracy_list, test_precision_list, test_recall_list)

    np.savez(out_path / "stats", train_loss=train_loss_list, train_accuracy=train_accuracy_list,
             train_precision=train_precision_list, train_recall=train_recall_list, test_loss=test_loss_list,
             test_accuracy=test_accuracy_list, test_precision=test_precision_list, test_recall=test_recall_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the chess model.")

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument("--gnn", action="store_true", help="GNN Model")
    group.add_argument("--mlp", action="store_true", help="MLP Model")

    parser.add_argument("--dataset-size", type=int, required=True, help="Size of the dataset to use.")
    parser.add_argument("--epochs", type=int, required=True, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, required=True, help="Batch size for training.")
    parser.add_argument("--load-workers", type=int, required=True, help="Number of data loading workers.")
    parser.add_argument("--model-save", type=Path, required=True, help="Path to save the trained model.")
    parser.add_argument("root", type=Path, help="Root directory of the dataset.")

    args = parser.parse_args()

    save_path = args.model_save
    save_path.mkdir(parents=True, exist_ok=True)
    (save_path / "keep_me").touch(exist_ok=True)

    train(args.root, args.dataset_size, args.epochs, args.batch_size, args.load_workers, save_path, args.gnn)
