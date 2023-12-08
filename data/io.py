from typing import Union, List
from pathlib import Path
from itertools import product
import numpy as np
import re
import os
import logging

from .sparsegraph import SparseGraph
from scipy.sparse import csr_matrix, diags

data_dir = Path(__file__).parent


def load_graphs(data_set: str, entity_name: str, version: str,
                directory: Union[Path, str] = data_dir
                ) -> SparseGraph:
    if isinstance(directory, str):
        directory = Path(directory)
    file_names = ['adjacency_matrices', 'feature_matrices', 'labels']
    if data_set == 'liar_dataset':
        return load_liar_graphs(directory / data_set, entity_name,
                                version, file_names)
    elif data_set == 'FakeNewsNet' and entity_name == 'author':
        return load_liar_graphs(directory / data_set, entity_name,
                                version, file_names)
    elif data_set == 'FakeNewsNet':
        return load_fnn_graphs(directory / data_set, entity_name,
                               version, file_names)


def load_liar_graphs(root_dir: Path, entity_name: str, version: str, file_names: List[str]):
    raw_graphs = {}
    for name in file_names:
        if name == 'labels':
            path_to_file = root_dir / Path(entity_name + '_' + name + '.npz')
        else:
            path_to_file = root_dir / \
                Path(entity_name + '_' + name + '_' + version + '.npz')
        if path_to_file.exists():
            with np.load(path_to_file, allow_pickle=True) as loader:
                raw_graphs[name] = dict(loader)
                if name != 'adjacency_matrices':
                    raw_graphs[name] = raw_graphs[name][name]
        else:
            raise ValueError("{} doesn't exist.".format(path_to_file))
    graphs = []
    for i, _ in enumerate(raw_graphs['labels']):
        adj_matrix = get_adjacency_matrix(
            {k: v[i] for k, v in raw_graphs['adjacency_matrices'].items()})
        attr_matrix = raw_graphs['feature_matrices'][i]
        label = raw_graphs['labels'][i]
        graphs.append(SparseGraph(adj_matrix, attr_matrix, label))
    return np.array(graphs)


def load_fnn_graphs(root_dir: Path, entity_name: str, version: str, file_names: List[str]):
    raw_graphs = {file_name: None for file_name in file_names}
    graphs = []
    with np.load(root_dir / Path(entity_name + '_' + file_names[-1] + '.npz'),
                 allow_pickle=True) as loader:
        raw_graphs[file_names[-1]] = dict(loader)[file_names[-1]]
    with np.load(root_dir / 'paths.npz', allow_pickle=True) as loader:
        paths = dict(loader)['paths']
    for i, path in enumerate(paths):
        for file_name in file_names[:2]:
            file = Path(entity_name + '_' + file_name +
                        '_' + version + '.npz')
            path_to_file = path / file
            if path_to_file.exists():
                with np.load(path_to_file, allow_pickle=True) as loader:
                    raw_graphs[file_name] = dict(loader)
                if file_name == 'feature_matrices':
                    raw_graphs[file_name] = raw_graphs[file_name][file_name]
                raw_graphs['exist'] = True
            else:
                raw_graphs['exist'] = False
                break
        if raw_graphs['exist']:
            adj_matrix = get_adjacency_matrix(
                raw_graphs['adjacency_matrices'])
            graphs.append(SparseGraph(adj_matrix,
                                      raw_graphs['feature_matrices'],
                                      raw_graphs['labels'][i]))
    return np.array(graphs)


def get_adjacency_matrix(adjacency_matrices):
    n = adjacency_matrices['adjacency_shape_list'][0]
    if n <= 1:
        adjacency_matrix = csr_matrix(np.zeros(shape=(n, n)))
    else:
        adjacency_matrix = diags(adjacency_matrices['adjacency_diagonals_list'],
                                 adjacency_matrices['adjacency_offset_list'],
                                 adjacency_matrices['adjacency_shape_list'])
    return adjacency_matrix


def load_mapping(name: str, directory: Union[Path, str] = data_dir) -> dict:
    if isinstance(directory, str):
        directory = Path(directory)
    if not name.endswith('.npz'):
        name += '.npz'
    path_to_file = directory / name
    with np.load(path_to_file, allow_pickle=True) as loader:
        mapping = dict(loader)['mapping']
    return mapping.transpose()
