import sys, logging
import numpy as np
import scipy.sparse as sp
import torch
import pickle
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
import random
from pathlib import Path

sys.path.insert(1, '../../data/script')
from News import News
from Author import Author
from Subject import Subject
from load_data import load_dataset


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

def load_mapping(name: str, directory: str) -> dict:
    directory = Path(directory)
    if not name.endswith('.npz'):
        name += '.npz'
    path_to_file = directory / name
    with np.load(path_to_file, allow_pickle=True) as loader:
        raw_mapping = dict(loader)['mapping']
    if len(raw_mapping) != 0:
        mapping = np.zeros(tuple(np.amax(raw_mapping, axis=0)+1))
        for row in raw_mapping:
            mapping[row[0], row[1]] = 1
        return mapping
    else:
        return np.empty(shape=(0,0))

def load_my_data(logger, data_set):
    root_dir = "../../data/" + data_set
    mapping_filenames = ['news_author', 'news_subject']
    adjs = []
    for name in mapping_filenames:
        mapping = load_mapping(name, root_dir)
        if len(mapping) != 0:
            adjs.append(mapping)

    news_dict, author_dict, subjects, _ = load_dataset(logger, data_set)
    contents, labels, _ = News.get_news_infos(news_dict)
    profiles, _ = Author.get_author_infos(author_dict)
    labels = encode_onehot(labels)
    vectorizer = TfidfVectorizer()
    subjects = Subject.get_subjects(subjects)
    features = []
    features.append(vectorizer.fit_transform(contents).todense())
    features.append(vectorizer.fit_transform(profiles).todense())
    if len(subjects) != 0:
        features.append(vectorizer.fit_transform(subjects).todense())
    # print(np.array(adjs).shape, np.array(features).shape, features[0].shape, 
    #       np.array(labels).shape)
    return adjs, features, labels, len(contents)

def load_data(path="../data/graph data/", dataset="fake news"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))
    features = []
    labels = []
    with open(path+'news_feature_vector_3000.pickle', 'rb') as f:
         feature = pickle.load(f)
         feature = normalize_features(feature)
         print(feature.shape, feature.todense()[:1,:5])
         features.append(feature.todense())
    f.close
    with open(path+'creator_feature_vector_3109.pickle', 'rb') as f:
         feature = pickle.load(f)
         feature = normalize_features(feature)
         print(feature.shape, feature.todense()[:1,:5])
         features.append(feature.todense())
    f.close
    with open(path+'subject_feature_vector_191.pickle', 'rb') as f:
         feature = pickle.load(f)
         feature = normalize_features(feature)
         print(feature.shape, feature.todense()[:1,:5])
         features.append(feature.todense())
    f.close
#    features = sp.csr_matrix(features, dtype=np.float32) 
    
    with open(path+'index_label_2_class.txt', 'r') as l:
        lines = l.readlines()
        for line in lines:
            line = line.split(' ')
            labels.append(int(line[1]))
    l.close
    labels = encode_onehot(labels)    
    print(labels.shape)

    adjs = []
    for adj_name in ['news_creator','news_subject']:
        with open(path+'{}_adj.pickle'.format(adj_name), 'rb') as f:
             adj = pickle.load(f) 
        f.close
#        adj = normalize_adj(adj + sp.eye(adj.shape[0]))
        adjs.append(adj.todense())
        print(adj.shape, adj.todense()[:2,:5])

    original = range(13826)
    idx_train = random.sample(original,2765)
    original = list(set(original) ^ set(idx_train))
    idx_val = random.sample(original,1000)
    original = list(set(original) ^ set(idx_val))
    idx_test = random.sample(original,2800)
    
    


    return adjs, features, labels, idx_train, idx_val, idx_test

def get_adjacency_matrix(adjacency_matrices):
    n = adjacency_matrices['adjacency_shape_list'][0]
    if n <= 1:
        adjacency_matrix = sp.csr_matrix(np.zeros(shape=(n, n)))
    else:
        adjacency_matrix = sp.diags(adjacency_matrices['adjacency_diagonals_list'],
                                 adjacency_matrices['adjacency_offset_list'],
                                 adjacency_matrices['adjacency_shape_list'])
    return adjacency_matrix

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def macro_f1(output, labels):
    preds = output.max(1)[1].type_as(labels)
    labels = labels.to(torch.device("cpu")).numpy()
    preds = preds.to(torch.device("cpu")).numpy()
    macro = metrics.f1_score(labels, preds, average='macro')  
    return macro

def micro_f1(output, labels):
    preds = output.max(1)[1].type_as(labels)
    labels = labels.to(torch.device("cpu")).numpy()
    preds = preds.to(torch.device("cpu")).numpy()
    micro = metrics.f1_score(labels, preds, average='micro')  
    return micro

def macro_precision(output, labels):
    preds = output.max(1)[1].type_as(labels)
    labels = labels.to(torch.device("cpu")).numpy()
    preds = preds.to(torch.device("cpu")).numpy()
    micro = metrics.precision_score(labels, preds, average='macro')  
    return micro

def macro_recall(output, labels):
    preds = output.max(1)[1].type_as(labels)
    labels = labels.to(torch.device("cpu")).numpy()
    preds = preds.to(torch.device("cpu")).numpy()
    micro = metrics.recall_score(labels, preds, average='macro')  
    return micro

def f1(output, labels):
    preds = output.max(1)[1].type_as(labels)
    labels = labels.to(torch.device("cpu")).numpy()
    preds = preds.to(torch.device("cpu")).numpy()
    macro = metrics.f1_score(labels, preds)  
    return macro


def precision(output, labels):
    preds = output.max(1)[1].type_as(labels)
    labels = labels.to(torch.device("cpu")).numpy()
    preds = preds.to(torch.device("cpu")).numpy()
    micro = metrics.precision_score(labels, preds)  
    return micro

def recall(output, labels):
    preds = output.max(1)[1].type_as(labels)
    labels = labels.to(torch.device("cpu")).numpy()
    preds = preds.to(torch.device("cpu")).numpy()
    micro = metrics.recall_score(labels, preds)  
    return micro