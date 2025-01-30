import json
import os
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Dataset, Data


class ChessData(Data):
    def __init__(self, num_nodes, x, edge_index, y, concat_x=True, **kwargs):
        super().__init__(num_nodes=num_nodes, x=x, edge_index=edge_index, y=y, **kwargs)

        self._new_dim = ['y']

        if not concat_x:
            self._new_dim.append('x')

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in self._new_dim:
            return None
        return super().__cat_dim__(key, value, *args, **kwargs)


class GraphChessDataset(Dataset):
    def __init__(self, root: str, cp_transform=None, concat_x=True, length=None, offset=None, **kwargs):
        super().__init__(root, **kwargs)
        self.concat_x = concat_x
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

    def _subdir(self, idx):
        return f"{idx % 255:02x}"

    def _nodes_chunk(self, idx):
        return self._nodes_dir / self._subdir(idx) / f"nodes_{idx}.npy"

    def _edges_chunk(self, idx):
        return self._edges_dir / self._subdir(idx) / f"edges_{idx}.npy"

    def get(self, idx):
        new_id = (idx + self._offset) if self._offset is not None else idx

        node = torch.from_numpy(np.load(self._nodes_chunk(new_id)).astype(np.float32))
        edges = torch.from_numpy(np.load(self._edges_chunk(new_id)).astype(np.long))
        
        raw_cp = self._cp_data[new_id]
        cp = self._transform_cp(raw_cp) if self._transform_cp is not None else raw_cp

        return ChessData(num_nodes=64, x=node, edge_index=edges, y=cp, concat_x=self.concat_x)


def build_train_test_dataset(root: str, dataset_size: int, train_prop: float = 0.9, test_prop: float = 0.1,
                             **kwargs) -> (GraphChessDataset, GraphChessDataset):
    if train_prop + test_prop > 1:
        raise ValueError("Proportions sum up to more than 1.")

    with open(os.path.join(root, "dataset.json"), 'r') as f:
        dataset_info = json.load(f)
        if dataset_size > dataset_info['size']:
            raise ValueError(f"The dataset is of size {dataset_info['size']}, cannot built one of {dataset_size}.")

    train_len = int(dataset_size * train_prop)
    test_len = int(dataset_size * test_prop)

    train_dataset = GraphChessDataset(root, length=train_len, offset=0, **kwargs)
    test_dataset = GraphChessDataset(root, length=test_len, offset=train_len, **kwargs)
    return train_dataset, test_dataset
