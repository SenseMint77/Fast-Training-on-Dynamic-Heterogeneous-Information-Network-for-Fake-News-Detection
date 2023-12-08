import sys, getopt, re, logging, random, time, torch, argparse
import numpy as np
from torch import tensor, transpose
from torch_geometric.data import Data
from torch_geometric.nn.models import GCN, GAT
import gensim.downloader as api

sys.path.insert(1, 'data/script')

from load_data import load_dataset
from News import News
from graph_utils import TOKEN_PATTERN
from pytorch_code.utils import get_K_fold_masks, get_train_val_mask
from pytorch_code.classifier import normal_train, normal_test

NUM_CLASSES = 2

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_set', type=str, required=True, help='Validate during training pass.')
parser.add_argument('-m', '--model', type=str, required=True, help='Validate during training pass.')
parser.add_argument('--lr_rate', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
parser.add_argument('--batch_size', type=int, default=128, help='Training and validation batch size')
parser.add_argument('--patience', type=int, default=10, help='Patience')
parser.add_argument('--n_hidden', type=int, default=64, help='The number of hidden nodes in a layer')
parser.add_argument('--n_layers', type=int, default=2, help='The number of layers in the model')


if __name__ == '__main__':
    # variables
    args = parser.parse_args()  
    with_test_author = False
    data_set = args.data_set
    model_type = args.model
    lr_rate = args.lr_rate
    weight_decay = args.weight_decay
    batch_size = args.batch_size
    patience = args.patience
    n_hidden = args.n_hidden
    n_layers = args.n_layers
    print(args)
    # load data
    log_filename = 'log/' + data_set + '_' + model_type + '_' + str(lr_rate) + \
                    '_' + str(weight_decay) + '_'+ str(batch_size) + \
                    '_' + str(n_hidden) + '_' + str(n_layers) + '_log.out'
    logging.basicConfig(filename=log_filename, filemode='w', 
                        format='%(asctime)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)
    news_dict, author_dict, subjects, sources = load_dataset(logger, data_set)
    contents, news_labels, paths = News.get_news_infos(news_dict)

    # get graph data
    pattern = re.compile("([\w][\w']*\w)")
    wv = api.load('glove-wiki-gigaword-50')
    data = np.empty(shape=(len(contents)), dtype=object)
    for idx, content in enumerate(contents):
        graph_data = {'x': []}
        content_arr = re.findall(TOKEN_PATTERN, content)
        for word in content_arr:
            if wv.has_index_for(word):
                graph_data['x'].append(wv[word])  
            else:
                graph_data['x'].append(np.zeros(shape=50))
        graph_data['x'] = np.array(graph_data['x'])
        graph_data['edge_index'] = []
        for i, _ in enumerate(graph_data['x']):
            graph_data['edge_index'] += [[i, j] for j in range(i-2, i+2) 
                                            if i != j and j >= 0 and j < len(content_arr)]
        data[idx] = Data(tensor(graph_data['x']),
                    transpose(tensor(graph_data['edge_index']), 0, 1),
                    y=tensor(news_labels[idx]))

    # set hyperparameters
    lr_rate = 0.01
    alpha = 0.05
    drop_prob = 0.5
    weight_decay = 0.01
    batch_size = 128
    # set other parameters
    print_interval = 20
    max_epochs = 1000
    K = 4

    train_test_masks = get_K_fold_masks(len(data), K, seed=random.randint(0, 99999))
    metrics = ['accuracy', 'precision', 'recall', 'auc', 'F1', 'loss']
    avg_results = {item: [] for item in metrics}
    logging.info("start training!")
    best_epochs = []
    start_time = time.time()
    for k in range(K):
        train_mask, test_mask = get_train_val_mask(
            train_test_masks, k)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # initialize model
        if model_type == 'gcn':
            model = GCN(data[0].x.shape[1], n_hidden, n_layers,
                        NUM_CLASSES, drop_prob).to(device)
        elif model_type == 'gat':
            model = GAT(data[0].x.shape[1], n_hidden, n_layers,
                        NUM_CLASSES, drop_prob).to(device)
        # set optimizer and loss criterion
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr_rate, weight_decay=weight_decay)
        criterion = torch.nn.CrossEntropyLoss()

        # train model
        bad_counter = 0
        epoch_results = {item: [] for item in metrics}
        best_epochs.append(0)
        
        for epoch in range(max_epochs):
            train_loss = normal_train(
                model, data[train_mask], optimizer, criterion, batch_size, device)
            test_results = normal_test(model, data[test_mask], batch_size, device)
            epoch_results['loss'].append(train_loss)
            for key in epoch_results.keys():
                if key != 'loss':
                    epoch_results[key].append(test_results[key])
            logging.info('Iter: %d/%d, Epoch: %03d, Train Loss: %.4f, Test Acc: %.4f',
                         k+1, K, epoch+1, train_loss, test_results['accuracy'])
            if train_loss < epoch_results['loss'][best_epochs[-1]]:
                best_epochs[-1] = epoch
                bad_counter = 0
            else:
                bad_counter += 1

            if bad_counter == patience:
                break
        for key in avg_results.keys():
            avg_results[key].append(epoch_results[key])
        logging.info('------------------')
    
    logging.info("Optimization Finished!")
    logging.info("Total time elapsed: {:.4f}s".format(time.time() - start_time))

    # evaluate average performance
    best_results = {}
    for key in avg_results.keys():
        if key != 'loss':
            best_results[key] = np.mean([avg_results[key][k][best_epochs[k]]
                                        for k in range(K)])
    logging.info("Avg accuracy: %0.4f", best_results['accuracy'])
    logging.info("    precision: %0.4f", best_results['precision'])
    logging.info("    recall: %0.4f", best_results['recall'])
    logging.info("    AUC: %0.4f", best_results['auc'])
    logging.info("    F1: %0.4f", best_results['F1'])

