import logging
import numpy as np
from pathlib import Path


def save_data(data_type, data_set, version, paths,
              adjacency_matrices, feature_matrices, labels):
    if data_set == 'liar_dataset':
        save_liar_graphs(paths[0], data_type, version, adjacency_matrices,
                         feature_matrices, labels)
    elif data_set == 'FakeNewsNet':
        save_fnn_graphs(paths, data_set, data_type, version, adjacency_matrices,
                        feature_matrices, labels)


def save_fnn_graphs(paths, data_set, data_type, version,
                    adjacency_matrices, feature_matrices, labels):
    path = str(paths[0])
    root_dir = Path(path[:path.find(data_set) + len(data_set)])
    with open(root_dir / Path(data_type + '_labels.npz'), 'wb') as f:
        np.savez(f, labels=labels)
    if data_type == 'news':
        for i, _ in enumerate(labels):
            adjacency_matrix = {k: v[i] for k, v in adjacency_matrices.items()}
            save_graphs(paths[i], data_type, version,
                        adjacency_matrix, feature_matrices[i])
            logging.info('i: %d, feature_matrix: %s, adj_matrix: {diagonals size: %s , '
                         'offset: %s, shape : %s}', i, feature_matrices[i].shape,
                         len(adjacency_matrices['adjacency_diagonals_list'][i]),
                         adjacency_matrices['adjacency_offset_list'][i],
                         adjacency_matrices['adjacency_shape_list'][i])
    else:
        save_graphs(root_dir, data_type, version,
                    adjacency_matrices, feature_matrices)


def save_liar_graphs(path_to_file, data_type, version,
                     adjacency_matrices, feature_matrices, labels):
    with open(path_to_file / Path(data_type + '_labels.npz'), 'wb') as f:
        np.savez(f, labels=labels)
    save_graphs(path_to_file, data_type, version,
                adjacency_matrices, feature_matrices)


def save_graphs(path_to_file, data_type, version, adjacency_matrices, feature_matrices):
    with open(path_to_file / Path(data_type + '_adjacency_matrices_' + version + '.npz'),
              'wb') as f:
        np.savez(f, **adjacency_matrices)
    with open(path_to_file / Path(data_type + '_feature_matrices_' + version + '.npz'),
              'wb') as f:
        np.savez(f, feature_matrices=feature_matrices)
