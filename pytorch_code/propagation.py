from typing import List
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import math
import random
import time

from .model_utils import MixedDropout, sparse_matrix_to_torch


def calc_A_hat(adj_matrix: sp.spmatrix) -> sp.spmatrix:
    nnodes = adj_matrix.shape[0]
    A = adj_matrix + sp.eye(nnodes)
    D_vec = np.sum(A, axis=1).A1
    D_vec_invsqrt_corr = 1 / np.sqrt(D_vec)
    D_invsqrt_corr = sp.diags(D_vec_invsqrt_corr)
    return D_invsqrt_corr @ A @ D_invsqrt_corr


def calc_ppr_exact(adj_matrix: sp.spmatrix, alpha: float) -> np.ndarray:
    nnodes = adj_matrix.shape[0]
    A_hat = calc_A_hat(adj_matrix)
    A_inner = sp.eye(nnodes) - (1 - alpha) * A_hat
    return alpha * np.linalg.inv(A_inner.toarray())


def calc_avg_ppr_exact(adj_matrices: List[sp.spmatrix], prop_weights: List[int], alpha: float) -> np.ndarray:
    avg_ppr = 0
    for i, adj_matrix in enumerate(adj_matrices):
        nnodes = adj_matrix.shape[0]
        A_hat = calc_A_hat(adj_matrix)
        A_inner = sp.eye(nnodes) - (1 - alpha) * A_hat
        avg_ppr += prop_weights[i] * alpha * np.linalg.inv(A_inner.toarray())
    return avg_ppr


def mask_adj_matrix(adj_matrix: sp.spmatrix) -> sp.spmatrix:
    nnodes = adj_matrix.shape[0]
    masked_n_edges = 100
    influenced_n_nodes = int(math.sqrt(masked_n_edges / 2))
    influenced_nodes = random.sample(range(1, nnodes), influenced_n_nodes)
    masked_adj_matrix = sp.lil_matrix(adj_matrix)
    for i in range(len(influenced_nodes)):
        for j in range(i + 1, len(influenced_nodes)):
            masked_adj_matrix[influenced_nodes[i], influenced_nodes[j]] = 0
            masked_adj_matrix[influenced_nodes[j], influenced_nodes[i]] = 0
    return masked_adj_matrix


def track_ppr(adj_matrix: sp.spmatrix, masked_adj_matrix: sp.spmatrix, ppr_mat, alpha):
    nnodes = adj_matrix.shape[0]

    A = masked_adj_matrix + sp.eye(nnodes)
    D_vec = np.sum(A, axis=1).A1
    D_vec_invsqrt_corr = 1 / np.sqrt(D_vec)
    D_invsqrt_corr = sp.diags(D_vec_invsqrt_corr)
    M = A @ D_invsqrt_corr

    A_prime = adj_matrix + sp.eye(nnodes)
    D_vec_prime = np.sum(A_prime, axis=1).A1
    D_vec_invsqrt_corr_prime = 1 / np.sqrt(D_vec_prime)
    D_invsqrt_corr_prime = sp.diags(D_vec_invsqrt_corr_prime)
    M_prime = A_prime @ D_invsqrt_corr_prime

    # --- push to approximate converged --- #
    diff_matrix = M_prime - M
    # - probability mass that needs to be pushed out - #
    pushout = alpha * diff_matrix @ ppr_mat.T
    acc_pushout = pushout                      # k = 0

    temp = alpha * M_prime                     # k = 1
    acc_pushout += temp @ pushout

    num_itr = 1                                # k starts from 2 to user-specified
    for k in range(num_itr):
        new_temp = temp * alpha @ M_prime
        acc_pushout += new_temp @ pushout
        temp = new_temp

    t_ppr = ppr_mat + acc_pushout
    # ------------------------------------ #

    return t_ppr


def track_2_hop_ppr(adj_matrix: sp.spmatrix, masked_adj_matrix: sp.spmatrix, ppr_mat, alpha):
    nnodes = adj_matrix.shape[0]

    A = masked_adj_matrix + sp.eye(nnodes)
    D_vec = np.sum(A, axis=1).A1
    D_vec_invsqrt_corr = 1 / np.sqrt(D_vec)
    D_invsqrt_corr = sp.diags(D_vec_invsqrt_corr)
    M = A @ D_invsqrt_corr
    M_2 = M @ M

    A_prime = adj_matrix + sp.eye(nnodes)
    D_vec_prime = np.sum(A_prime, axis=1).A1
    D_vec_invsqrt_corr_prime = 1 / np.sqrt(D_vec_prime)
    D_invsqrt_corr_prime = sp.diags(D_vec_invsqrt_corr_prime)
    M_prime = A_prime @ D_invsqrt_corr_prime
    M_2_prime = M_prime @ M_prime

    # --- push to approximate converged --- #
    diff_matrix = M_2_prime - M_2
    # - probability mass that needs to be pushed out - #
    pushout = alpha * diff_matrix @ ppr_mat.T
    acc_pushout = pushout                      # k = 0

    temp = alpha * M_2_prime                     # k = 1
    acc_pushout += temp @ pushout

    num_itr = 1                                # k starts from 2 to user-specified
    for k in range(num_itr):
        new_temp = temp * alpha @ M_2_prime
        acc_pushout += new_temp @ pushout
        temp = new_temp

    t_ppr = ppr_mat + acc_pushout
    # ------------------------------------ #

    return t_ppr


def track_mixed_ppr(adj_matrices: List[sp.spmatrix], masked_adj_matrices: List[sp.spmatrix],
                    prop_weights, ppr_mat, alpha):
    M_total = np.empty(shape=adj_matrices[0].shape)
    for i, adj_matrix in enumerate(adj_matrices):
        masked_adj_matrix = masked_adj_matrices[i]
        A = masked_adj_matrix + sp.eye(adj_matrix.shape[0])
        D_vec = np.sum(A, axis=1).A1
        D_vec_invsqrt_corr = 1 / np.sqrt(D_vec)
        D_invsqrt_corr = sp.diags(D_vec_invsqrt_corr)
        M = A @ D_invsqrt_corr
        if i == 0:
            M_2 = M @ M
            M_total += prop_weights[i]*M_2
        else:
            M_total += prop_weights[i]*M

    M_prime_total = np.empty(shape=adj_matrices[0].shape)
    for i, adj_matrix in enumerate(adj_matrices):
        A_prime = adj_matrix + sp.eye(adj_matrix.shape[0])
        D_vec_prime = np.sum(A_prime, axis=1).A1
        D_vec_invsqrt_corr_prime = 1 / np.sqrt(D_vec_prime)
        D_invsqrt_corr_prime = sp.diags(D_vec_invsqrt_corr_prime)
        M_prime = A_prime @ D_invsqrt_corr_prime
        if i == 0:
            M_2_prime = M_prime @ M_prime
            M_prime_total += prop_weights[i]*M_2_prime
        else:
            M_prime_total += prop_weights[i]*M_prime

    # --- push to approximate converged --- #
    diff_matrix = M_prime_total - M_total
    # - probability mass that needs to be pushed out - #
    pushout = alpha * diff_matrix @ ppr_mat.T
    acc_pushout = pushout                      # k = 0

    temp = alpha * M_prime_total                     # k = 1
    acc_pushout += temp @ pushout

    num_itr = 1                                # k starts from 2 to user-specified
    for k in range(num_itr):
        new_temp = temp * alpha @ M_prime_total
        acc_pushout += new_temp @ pushout
        temp = new_temp

    t_ppr = ppr_mat + acc_pushout
    # ------------------------------------ #

    return t_ppr


class PPRExact(nn.Module):
    def __init__(self, adj_matrix: sp.spmatrix, alpha: float, drop_prob: float = None):
        super().__init__()

        ppr_mat = calc_ppr_exact(adj_matrix, alpha)
        self.register_buffer('mat', torch.FloatTensor(ppr_mat))

        if drop_prob is None or drop_prob == 0:
            self.dropout = lambda x: x
        else:
            self.dropout = MixedDropout(drop_prob)

    def forward(self, predictions: torch.FloatTensor, idx: torch.LongTensor):
        # - aggregating neighbourhood predictions - #
        return self.dropout(self.mat[idx]) @ predictions


class PPRPowerIteration(nn.Module):
    def __init__(self, adj_matrix: sp.spmatrix, alpha: float, niter: int, drop_prob: float = None):
        super().__init__()
        self.alpha = alpha
        self.niter = niter

        M = calc_A_hat(adj_matrix)
        self.register_buffer('A_hat', sparse_matrix_to_torch((1 - alpha) * M))

        if drop_prob is None or drop_prob == 0:
            self.dropout = lambda x: x
        else:
            self.dropout = MixedDropout(drop_prob)

    def forward(self, local_preds: torch.FloatTensor, idx: torch.LongTensor):
        preds = local_preds
        for _ in range(self.niter):
            A_drop = self.dropout(self.A_hat)
            preds = A_drop @ preds + self.alpha * local_preds
        return preds[idx]


class SDG(nn.Module):
    def __init__(self, adj_matrix: sp.spmatrix, alpha: float, drop_prob: float = None):
        super().__init__()

        start_time = time.time()

        # last time graph structure and its ppr matrix
        masked_adj_matrix = mask_adj_matrix(adj_matrix)
        ppr_mat = calc_ppr_exact(masked_adj_matrix, alpha)

        print('Generating the new graph costs: ' +
              str(time.time() - start_time) + ' sec.')

        # tracked ppr matrix
        t_ppr = track_ppr(adj_matrix, masked_adj_matrix, ppr_mat, alpha)
        self.register_buffer('mat', torch.FloatTensor(t_ppr))

        if drop_prob is None or drop_prob == 0:
            self.dropout = lambda x: x
        else:
            self.dropout = MixedDropout(drop_prob)

    def forward(self, predictions: torch.FloatTensor, idx: torch.LongTensor):
        return self.dropout(self.mat[idx]) @ predictions


class TwoHopDynamicPropagation(nn.Module):
    def __init__(self, logger, adj_matrices: List[sp.spmatrix],
                 alpha: float, drop_prob: float = None) -> None:
        super().__init__()

        start = time.time()
        # last time graph structure and its ppr matrix
        adj_matrix = adj_matrices[0]
        masked_adj_matrix = mask_adj_matrix(adj_matrix)
        ppr_mat = calc_ppr_exact(masked_adj_matrix, alpha)
        logger.info('Generating the new graph costs: ' +
                     str(time.time() - start) + ' sec.')
        # tracked ppr matrix
        t_ppr = track_2_hop_ppr(adj_matrix, masked_adj_matrix, ppr_mat, alpha)
        self.register_buffer('mat', torch.FloatTensor(t_ppr))

        if drop_prob is None or drop_prob == 0:
            self.dropout = lambda x: x
        else:
            self.dropout = MixedDropout(drop_prob)

    def forward(self, predictions: torch.FloatTensor, idx: torch.LongTensor):
        return self.dropout(self.mat[idx]) @ predictions


class MixedDynamicPropagation(nn.Module):
    def __init__(self, logger, adj_matrices: List[sp.spmatrix], prop_weights: List[int],
                 alpha: float, drop_prob: float = None) -> None:
        super().__init__()

        start = time.time()
        # last time graph structure and its ppr matrix
        masked_adj_matrices = []
        for adj_matrix in adj_matrices:
            masked_adj_matrices.append(mask_adj_matrix(adj_matrix))
        avg_ppr_mat = calc_avg_ppr_exact(adj_matrices, prop_weights, alpha)
        logger.info('Generating the new graph costs: ' +
                     str(time.time() - start) + ' sec.')
        # tracked ppr matrix
        t_ppr = track_mixed_ppr(
            adj_matrices, masked_adj_matrices, prop_weights, avg_ppr_mat, alpha)
        self.register_buffer('mat', torch.FloatTensor(t_ppr))

        if drop_prob is None or drop_prob == 0:
            self.dropout = lambda x: x
        else:
            self.dropout = MixedDropout(drop_prob)

    def forward(self, predictions: torch.FloatTensor, idx: torch.LongTensor):
        return self.dropout(self.mat[idx]) @ predictions
