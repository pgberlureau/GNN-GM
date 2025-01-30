import json
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Dataset, Data
from torch_geometric.typing import OptTensor


class ChessData(Data):
    def __init__(self, num_nodes, x, edge_index, y, new_dim=None, **kwargs):
        super().__init__(num_nodes=num_nodes, x=x, edge_index=edge_index, y=y, **kwargs)
        self._new_dim = set() if new_dim is None else new_dim
        self._new_dim.add('y')

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in self._new_dim:
            return None
        return super().__cat_dim__(key, value, *args, **kwargs)


class GraphChessDataset(Dataset):
    def __init__(self, root: str, transform=None, cp_transform=None, length=None, offset=None, new_dim=None):
        super().__init__(root, transform)
        self.new_dim = new_dim
        self._offset = offset
        self._transform_cp = cp_transform

        with open(os.path.join(root, "dataset.json"), 'r') as f:
            dataset_info = json.load(f)

            self._len = length if length is not None else dataset_info['size']
            self._chunk_size = dataset_info['chunk_size']
            self._edges_dir = Path(root) / dataset_info['edges_dir']
            self._nodes_dir = Path(root) / dataset_info['nodes_dir']
            self._cp_data = torch.from_numpy(np.load(os.path.join(self.root, f"cp.npy")))

    def len(self):
        return self._len

    def _nodes_chunk(self, chunk_id):
        return self._nodes_dir / f"nodes_{chunk_id}.npz"

    def _edges_chunk(self, chunk_id):
        return self._edges_dir / f"edges_{chunk_id}.npz"

    def get(self, idx):
        new_id = (idx + self._offset) if self._offset is not None else idx
        chunk_id = new_id // self._chunk_size
        chunk_index = new_id % self._chunk_size

        nodes_chunk = np.load(self._nodes_chunk(chunk_id))
        edges_chunk = np.load(self._edges_chunk(chunk_id))

        node = torch.from_numpy(nodes_chunk[f"node_{chunk_index}"].astype(np.float32))
        edges = torch.from_numpy(edges_chunk[f"edges_{chunk_index}"].astype(np.long))
        raw_cp = self._cp_data[new_id]
        cp = self._transform_cp(raw_cp) if self._transform_cp is not None else raw_cp

        return ChessData(num_nodes=64, x=node, edge_index=edges, y=cp, new_dim=self.new_dim)
