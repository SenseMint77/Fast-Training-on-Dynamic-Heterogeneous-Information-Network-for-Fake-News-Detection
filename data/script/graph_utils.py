import re
import numpy as np
from unicodedata import normalize
from scipy.sparse.base import spmatrix

TOKEN_PATTERN = '(?ui)\\b\\w*[a-z]+\\w*\\b'


def get_feature_matrices(corpus, embed_corpus, token_indices) -> np.ndarray:
    feature_matrices = []
    for i, _ in enumerate(corpus):
        k = 0
        buf = ''
        embed_vector = []
        feature_matrix = []
        # convert sentence into word array, ignore numbers
        text_arr = re.findall(TOKEN_PATTERN, corpus[i])
        for j, _ in enumerate(token_indices[i]):
            # ignore paddings upon the end of sentence
            if k == len(text_arr):
                break
            # print(token_indices[i][j][0], text_arr[k])
            # ignore bert start and end token encodings
            if token_indices[i][j][0] == '[CLS]' or token_indices[i][j][0] == '[SEP]':
                continue
            # normalise special characters
            text_arr[k] = normalize('NFD', text_arr[k]).encode(
                'ascii', 'ignore').decode()
            # combine embeddings of substrings
            if not re.match(buf + re.escape(token_indices[i][j][0].replace('##', '')),
                            text_arr[k].casefold()):
                buf = ''
                continue
            buf += token_indices[i][j][0].replace('##', '')
            embed_vector.append(embed_corpus[i][j])
            # get aggregate embedding vector for k-th word
            if buf == text_arr[k].casefold():
                if len(embed_vector) > 1:
                    feature_matrix.append(np.mean(embed_vector, axis=0))
                else:
                    feature_matrix.append(embed_vector[0])
                buf = ''
                embed_vector = []
                k += 1
        feature_matrices.append(np.array(feature_matrix))
    return feature_matrices


def get_adjacency_matrices(corpus, feature_matrices, window_size=4) -> spmatrix:
    diagonals_list = []
    offset_list = []
    shape_list = []
    # get row and col indices for diags sparse matrix construction
    diag_mat_id = [i for i in range(1, window_size)]
    diag_mat_id += [-i for i in range(1, window_size)]
    diagonals = np.full(fill_value=1, shape=(len(diag_mat_id)))
    for idx, _ in enumerate(corpus):
        feature_length = len(feature_matrices[idx])
        shape = (feature_length, feature_length)
        mat_half_size = min(window_size, feature_length)
        offset = [i for i in range(1, mat_half_size)]
        offset += [-i for i in range(1, mat_half_size)]
        diagonals = np.full(fill_value=1, shape=len(offset))
        diagonals_list.append(diagonals)
        offset_list.append(offset)
        shape_list.append(shape)
    adjacency_matrices = {
        'adjacency_diagonals_list': diagonals_list,
        'adjacency_offset_list': offset_list,
        'adjacency_shape_list': shape_list,
    }
    return adjacency_matrices
