import json
import os

import numpy as np
import torch
from torch_geometric.data import Dataset, Data


class ChessData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ['x', 'y']:
            return None
        return super().__cat_dim__(key, value, *args, **kwargs)


class GraphChessDataset(Dataset):
    def __init__(self, root: str, transform=None, cp_transform=None):
        super().__init__(root, transform)

        with open(os.path.join(root, "dataset.json"), 'r') as f:
            dataset_info = json.load(f)
            print(f"We found: {dataset_info}")

            self._nb_chunk = dataset_info['nb_chunk']
            self._len = dataset_info['size']
            self._chunk_size = dataset_info['chunk_size']

            cp_data = np.load(os.path.join(self.root, f"cp.npy"))
            self._cp_data = cp_transform(cp_data) if cp_transform is not None else cp_data

    def len(self):
        return self._len

    def _nodes_chunk(self, chunk_id):
        return os.path.join(self.root, f"nodes_{chunk_id}.npz")

    def _edges_chunk(self, chunk_id):
        return os.path.join(self.root, f"edges_{chunk_id}.npz")

    def get(self, idx):
        chunk_id = idx // self._chunk_size
        nodes_chunk = np.load(self._nodes_chunk(chunk_id))
        edges_chunk = np.load(self._edges_chunk(chunk_id))

        node = torch.from_numpy(nodes_chunk[f"node_{idx}"].astype(np.float32))
        edges = torch.from_numpy(edges_chunk[f"edges_{idx}"].astype(np.long))

        return Data(num_nodes=64, x=node, edge_index=edges, y=self._cp_data[idx])
