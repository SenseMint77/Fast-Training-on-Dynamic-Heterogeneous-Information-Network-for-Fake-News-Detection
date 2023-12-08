from re import M
import numpy as np
import random
from torch import tensor, transpose
from torch_geometric.data import Data
from math import ceil
from scipy.sparse import csr_matrix
from itertools import product


def get_graph_data(data_set, graph_embeds, mappings, labels):
    '''
    graph data split for transductive learning
    '''
    data = {'x': graph_embeds['news'], 'edge_indices': [],
            'y': labels['news']}
    news_length = len(graph_embeds['news'])
    data['mask'] = np.arange(news_length)
    for i, mapping in enumerate(mappings):
        if len(mapping) != 0:
            add_mapping_data(data, mapping, graph_embeds, labels,
                             map(i)['from'], map(i)['to'], data_set)
    data = Data(tensor(data['x']),
                transpose(tensor(np.concatenate(data['edge_indices'])), 0, 1),
                y=tensor(data['y']), adj_matrices=get_adj_matrices(data))
    return data


def mask_graph_data(graph_data, mask, with_test_author=True):
    # add other information to mask
    graph_data.train_mask = mask['train']
    graph_data.test_mask = mask['test']
    news_length = len(mask['train']) + len(mask['test'])
    graph_data.train_mask = np.concatenate(
        (graph_data.train_mask, np.arange(news_length, graph_data.x.shape[0])))
    if with_test_author:
        graph_data.test_mask = np.concatenate(
            (graph_data.test_mask, np.arange(news_length, graph_data.x.shape[0])))
    return graph_data


def add_mapping_data(group, mapping, graph_embeds, labels, src_map, dst_map, data_set):
    if src_map == 'news' and dst_map == 'author':
        return add_news_author_mapping_data(group, mapping, graph_embeds[dst_map],
                                            labels[dst_map])
    else:
        homogeneous_mapping = get_homogeneous_mapping(mapping.transpose(), src_map, data_set)
        return add_other_mapping_data(group, homogeneous_mapping, src_map)


def get_homogeneous_mapping(mapping, src, data_set):
    combined_mapping = {}
    for row in mapping:
        if row[1] in combined_mapping:
            combined_mapping[row[1]].append(row[0])
        else:
            combined_mapping[row[1]] = [row[0]]
    all_homogeneous_mapping = {}
    homogeneous_mapping = []
    for _, item in combined_mapping.items():
        for row in product(item, item):
            if row[0] != row[1]:
                if (row[0], row[1]) in all_homogeneous_mapping.keys():
                    all_homogeneous_mapping[(row[0], row[1])] += 1
                else:
                    all_homogeneous_mapping[(row[0], row[1])] = 1
    for key, item in all_homogeneous_mapping.items():
        if data_set == 'liar_dataset':
            if src == 'news' and item >= 3:
                homogeneous_mapping.append([key[0], key[1]]) 
            elif src == 'author' and item >= 2:
                homogeneous_mapping.append([key[0], key[1]]) 
        else:
            homogeneous_mapping.append([key[0], key[1]])
    # print(f'dataset: {data_set}, #{src}-{src}: {len(homogeneous_mapping)}')
    return homogeneous_mapping


def add_other_mapping_data(group, mapping, src_map):
    edge_index = []
    if src_map == 'news':
        for row in mapping:
            if not row[0] in group['mask'] or not row[1] in group['mask']:
                continue
            src_idx = np.where(group['mask'] == row[0])[0][0]
            dst_idx = np.where(group['mask'] == row[1])[0][0]
            edge_index.append([src_idx, dst_idx])
    else:
        for row in mapping:
            if not row[0] in group['author_map'].keys() \
                    or not row[1] in group['author_map'].keys():
                continue
            src_idx = group['author_map'][row[0]]
            dst_idx = group['author_map'][row[1]]
            edge_index.append([src_idx, dst_idx])
    if len(edge_index) == 0:
        group['edge_indices'].append(np.empty(shape=(0, 2)))
    else:
        group['edge_indices'].append(edge_index)
    return group


def add_news_author_mapping_data(group, mapping, graph_embeds, labels):
    '''append corresponding mapping data into each group'''
    group_author_ids = {}
    group['edge_indices'].append([])
    for idx, news_id in enumerate(group['mask']):
        if news_id not in mapping[0]:
            continue
        author_ids = [mapping[1][i]
                      for i, val in enumerate(mapping[0]) if val == news_id]
        for author_id in author_ids:
            if author_id not in group_author_ids.keys():
                group['x'] = np.vstack((group['x'], graph_embeds[author_id]))
                group['y'] = np.append(group['y'], labels[author_id])
                group_author_ids[author_id] = len(group['x']) - 1
                group['edge_indices'][-1].append([idx, len(group['x']) - 1])
            else:
                idy = group_author_ids[author_id]
                group['edge_indices'][-1].append([idx, idy])
    group['author_map'] = group_author_ids
    return group


def map(idx):
    return {
        0: {
            'from': 'news',
            'to': 'author',
        },
        1: {
            'from': 'news',
            'to': 'news',
        },
        2: {
            'from': 'author',
            'to': 'author'
        }
    }[idx]


def get_adj_matrices(group):
    adj_matrices = []
    for edge_index in group['edge_indices']:
        adj_mat_data = np.full(shape=(len(edge_index[0])), fill_value=1)
        adj_matrix = csr_matrix(
            (adj_mat_data, (edge_index[0], edge_index[1])),
            shape=(group['x'].shape[0], group['x'].shape[0]))
        adj_matrices.append(adj_matrix)
    return adj_matrices


def get_random_mask(length, ratio, seed=1234):
    n_train = round(ratio[0]*length)
    n_val = round(ratio[1]*length)
    n_test = length - (n_train + n_val)
    random_indices = np.arange(length)
    print('seed: ', seed)
    random.Random(seed).shuffle(random_indices)
    train_mask = random_indices[:n_train]
    val_mask = random_indices[n_train:n_train+n_val]
    test_mask = random_indices[n_train+n_val:]
    return train_mask, val_mask, test_mask


def partition(obj, mask1, mask2, mask3=None):
    if mask3 is None:
        return np.array(obj[mask1]), np.array(obj[mask2])
    else:
        return np.array(obj[mask1]), np.array(obj[mask2]), np.array(obj[mask3])


def get_K_fold_masks(length, K, seed=1234):
    step = ceil(length/K)
    random_indices = np.arange(length)
    random.Random(seed).shuffle(random_indices)
    masks = [list(random_indices[i:i + step])
             for i in range(0, length, step)]
    return masks


def get_train_val_mask(masks, val_id):
    val_mask = masks[val_id]
    train_mask = []
    for j in range(val_id):
        train_mask += masks[j]
    for j in range(val_id+1, len(masks)):
        train_mask += masks[j]
    return train_mask, val_mask
