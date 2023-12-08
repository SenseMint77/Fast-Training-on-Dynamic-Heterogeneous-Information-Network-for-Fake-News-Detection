from __future__ import division
from __future__ import print_function

import os, sys, logging
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

from utils import load_data, load_my_data, accuracy, macro_f1, micro_f1, macro_precision, macro_recall,f1, precision, recall
from models import HGAT

sys.path.insert(1, '../../')
from pytorch_code.utils import get_K_fold_masks, get_train_val_mask


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('-data_set', type=str, default='liar_dataset')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--nhidden', type=int, default=10, help='Number of node level hidden units.')
parser.add_argument('--shidden', type=int, default=8, help='Number of semantic level hidden units.')
parser.add_argument('--nb_heads', type=int, default=1, help='Number of head attentions.')
parser.add_argument('--nd_dropout', type=float, default=0.4, help='Dropout rate (1 - keep probability).')
parser.add_argument('--se_dropout', type=float, default=0.0, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.1, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=10, help='Patience')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
if torch.cuda.is_available():
     device = torch.device("cuda")
     torch.cuda.set_device(0)
else:
     device = torch.device("cpu")
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
data_set = args.data_set
log_filename = 'log/'+ data_set + '_HGAT_' + str(args.lr) + '_' + str(args.weight_decay) + \
                '_' + str(args.nhidden) + '_' + str(args.shidden) + '_' + str(args.nb_heads) + \
                '_' + str(args.nd_dropout) + '_' + str(args.se_dropout) + '_' + str(args.alpha) + \
                '_log.out'
print(log_filename)
logging.basicConfig(filename=log_filename, 
                    filemode='w', format='%(asctime)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
# adjs, features, labels, idx_train, idx_val, idx_test = load_data()
adjs, features, labels, n_news = load_my_data(logger, data_set)

t_feat = features[0].shape[1]
nfeat_list = []
for i in range(1,len(features)):
    nfeat_list.append(features[i].shape[1])

for a in range(len(adjs)):
    adjs[a] = torch.FloatTensor(adjs[a])
    if args.cuda:
        adjs[a] = adjs[a].cuda()
for f in range(len(features)):
    features[f] = torch.FloatTensor(features[f])  
    if args.cuda:
        features[f] = features[f].cuda()
labels = torch.LongTensor(np.where(labels)[1])

# Model and optimizer
K = 4
train_test_masks = get_K_fold_masks(n_news, K, seed=random.randint(0, 99999))
metrics = ['accuracy', 'precision', 'recall', 'auc', 'F1']
avg_results = {key: [] for key in metrics}
logger.info("start training!")
start_time = time.time()

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adjs)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adjs)

    true_labels = labels[idx_val].cpu()
    pred_labels = output[idx_val].argmax(dim=1).cpu()
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])

    result = {}
    # print(confusion_matrix(true_labels, pred_labels, labels=[0,1]))
    result['accuracy'] = accuracy_score(true_labels, pred_labels)
    result['precision'] = precision_score(true_labels, pred_labels)
    result['recall'] = recall_score(true_labels, pred_labels)
    result['auc'] = roc_auc_score(true_labels, pred_labels)
    result['F1'] = f1_score(true_labels, pred_labels)
    logger.info('Epoch: %.4d, loss_train: %.4f, acc_train: %.4f,'\
                 'loss_val: %.4f, acc_val: %.4f', 
                 epoch+1, loss_train.item(), acc_train.item(), 
                 loss_val.item(), result['accuracy'])
    return loss_val.item(),loss_train.item(), result

def compute_test(idx_test):
    model.eval()
    output = model(features, adjs)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    mac = f1(output[idx_test], labels[idx_test])  
    mac_pre = precision(output[idx_test], labels[idx_test])  
    mac_rec = recall(output[idx_test], labels[idx_test])   
    logger.info("Test set results:",
                 "loss= {:.4f}".format(loss_test.item()),
                 "accuracy= {:.4f}".format(acc_test.item()),
                 "macro_f1= {:.4f}".format(mac),
                 "macro_precision= {:.4f}".format(mac_pre),
                 "macro_recall= {:.4f}".format(mac_rec))

for k in range(K):
    val_results = {key: [] for key in metrics}
    idx_train, idx_val = get_train_val_mask(train_test_masks, k)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    model = HGAT(tfeat = t_feat,
                nfeat_list=nfeat_list, 
                nhid=args.nhidden, 
                shid=args.shidden,
                nclass=int(labels.max()) + 1, 
                nd_dropout=args.nd_dropout,
                se_dropout=args.se_dropout,
                nheads=args.nb_heads, 
                alpha=args.alpha)
    optimizer = optim.Adam(model.parameters(), 
                        lr=args.lr, 
                        weight_decay=args.weight_decay)

    if args.cuda:
        model.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()

    # Train model
    loss_values = []
    loss_values_output = []
    bad_counter = 0
    best = args.epochs + 1
    best_epoch = 0
    for epoch in range(args.epochs):
        val_loss, train_loss, test_result = train(epoch)
        loss_values.append(train_loss)
        loss_values_output.append(val_loss)
        for key in test_result.keys():
            val_results[key].append(test_result[key])
        torch.save(model.state_dict(), '{}.pkl'.format(epoch))
        if loss_values[-1] < best:
            best = loss_values[-1]
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            break

        files = glob.glob('*.pkl')
        for file in files:
            epoch_nb = int(file.split('.')[0])
            if epoch_nb < best_epoch:
                os.remove(file)

    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb > best_epoch:
            os.remove(file)
    for key in avg_results.keys():
        avg_results[key].append(val_results[key])

logger.info("Optimization Finished!")
logger.info("Total time elapsed: %.4f", time.time() - start_time)

# evaluate average performance
logger.info("Avg accuracy: %.4f", np.mean(
    avg_results['accuracy'][:][-1]))
logger.info("    precision: %.4f", np.mean(
    avg_results['precision'][:][-1]))
logger.info("    recall: %.4f", np.mean(avg_results['recall'][:][-1]))
logger.info("    AUC: %.4f", np.mean(avg_results['auc'][:][-1]))
logger.info("    F1: %.4f", np.mean(avg_results['F1'][:][-1]))

# Restore best model
# print('Loading {}th epoch'.format(best_epoch))
# model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))
#for name, param in model.named_parameters():
#    if param.requires_grad:
#        print(name)

# Testing
# compute_test(idx_test)
#print(len(loss_values_output))
#print(loss_values_output)
#print(len(loss_values))
#print(loss_values)