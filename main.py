import sys, getopt, logging, time, torch, argparse
import numpy as np

from data.io import load_graphs, load_mapping
from pytorch_code.propagation import MixedDynamicPropagation, TwoHopDynamicPropagation
from pytorch_code.propagation import calc_ppr_exact
from pytorch_code.model import agnostic_model
from pytorch_code.classifier import train, test
from pytorch_code.utils import get_K_fold_masks, get_train_val_mask
from pytorch_code.utils import get_graph_data, mask_graph_data
import matplotlib.pyplot as plt

NUM_CLASSES = 2
NUM_HIDDEN = 64

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_set', type=str, required=True, help='Validate during training pass.')
parser.add_argument('-m', '--model', type=str, required=True, help='Validate during training pass.')
parser.add_argument('-v', '--version', type=str, required=True, help='Validate during training pass.')
parser.add_argument('--propagation_weights', type=float, nargs='+', default=[0.4,0.3,0.3])
parser.add_argument('--lr_rate', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
parser.add_argument('--drop_prob', type=float, default=0.0, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.8, help='Alpha')
parser.add_argument('--beta', type=float, default=0.1, help='Beta')
parser.add_argument('--patience', type=int, default=10, help='Patience')

if __name__ == '__main__':
    # set parameters from args
    args = parser.parse_args()
    lr_rate = args.lr_rate
    alpha = args.alpha
    drop_prob = args.drop_prob
    weight_decay = args.weight_decay
    propagation_weights = args.propagation_weights
    patience = args.patience
    data_set = args.data_set
    model_type = args.model
    version = args.version
    beta = args.beta
    print(args)
    # set other parameters
    max_epochs = 1000
    K = 4
    epsilon = 1e-3
    entity_names = ['news', 'author']
    if model_type == 'bipartite':
        mapping_names = ['news_author']
    else:
        mapping_names = ['news_author', 'news_subject', 'author_source']

    log_filename = 'log/' + data_set + '_' + model_type + '_' + version + \
                    str(lr_rate) + str(alpha) + str(beta) + str(drop_prob) + str(weight_decay) + \
                    str(propagation_weights).replace(' ', '') + '_log.out'
    logging.basicConfig(filename=log_filename, filemode='w', 
                        format='%(asctime)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)
    # change to True to include author info during test
    with_test_author = False

    graph_embeds = {
        entity_names[0]: [],
        entity_names[1]: []
    }
    mappings = []
    labels = {
        entity_names[0]: [],
        entity_names[1]: []
    }

    # aggregate graph data
    for name in entity_names:
        logging.info('processing %s', name)
        graphs = load_graphs(data_set, name, version)
        for graph in graphs:
            # get the Personalized PageRank of each node in graph
            n = graph.adj_matrix.shape[0]
            ppr_mat = calc_ppr_exact(graph.adj_matrix, beta)
            relative_weights = ppr_mat @ np.identity(n)

            # get aggregate node feature embedding from
            # H = [other nodes' embeddings] x [relative_weights]
            H = relative_weights @ graph.attr_matrix

            # mean pool to get graph embedding
            graph_embed = np.mean(H, axis=0)
            graph_embeds[name].append(graph_embed)
            labels[name].append(graph.label)
        graph_embeds[name] = np.array(graph_embeds[name])
        labels[name] = np.array(labels[name])
    for name in mapping_names:
        mappings.append(load_mapping(data_set + '/' + name))

    train_test_masks = get_K_fold_masks(len(graph_embeds[entity_names[0]]), K)
    metrics = ['accuracy', 'precision', 'recall', 'auc', 'F1', 'loss']
    avg_results = {item: [] for item in metrics}
    data = get_graph_data(data_set, graph_embeds, mappings, labels)
    # Perform K-fold cross validation
    logging.info("start training!")
    best_epochs = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # initialize dynamic propagation
    start_time = time.time()
    if model_type == 'bipartite':
        prop = TwoHopDynamicPropagation(
            logger, data.adj_matrices, alpha, drop_prob).to(device)
    else:
        prop = MixedDynamicPropagation(
            logger, data.adj_matrices, propagation_weights, alpha, drop_prob).to(device)
    prop_time = time.time() - start_time
    start_time = time.time()
    for k in range(K):
        train_mask, test_mask = get_train_val_mask(train_test_masks, k)
        mask = {'train': train_mask, 'test': test_mask}
        data = mask_graph_data(data, mask, with_test_author=with_test_author)
        logging.info("data mask split [train %s, test %s]",
                     len(data.train_mask), len(data.test_mask))
        # initialize model
        model = agnostic_model(data.x.shape[1], NUM_CLASSES, [NUM_HIDDEN],
                               drop_prob, prop).to(device)
        # set optimizer and loss criterion
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr_rate, weight_decay=weight_decay)
        criterion = torch.nn.CrossEntropyLoss()

        # train model
        current_val_acc = 1
        bad_counter = 0
        epoch_results = {item: [] for item in metrics}
        best_epochs.append(0)
        for epoch in range(max_epochs):
            train_loss, train_acc = train(model, data, optimizer, criterion, device)
            train_loss = train_loss.detach().numpy()
            test_results = test(model, data, device)
            epoch_results['loss'].append(train_loss)
            for key in epoch_results.keys():
                if key != 'loss':
                    epoch_results[key].append(test_results[key])
            logging.info('Iter: %d/%d, Epoch: %03d, Train Loss: %.4f, '\
                         'Train Acc: %.4f, Test Acc: %.4f', k+1, K,
                         epoch+1, train_loss, train_acc, test_results['accuracy'])
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
    logging.info("Propagation time elapsed: {:4f}s".format(prop_time))

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

    # plot training losses and test accuracy
    fig, ax = plt.subplots(figsize=(8, 5))
    for k in range(K):
        x_len = len(avg_results['loss'][k])
        ax.plot(np.arange(1, x_len+1), avg_results['loss'][k],
                label=f'k = {k}')
    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')
    ax.set_title('Training Loss')
    ax.legend()
    file_name = 'loss/' + data_set + '_' + model_type + '_' + version + \
                str(lr_rate) + str(alpha) + str(drop_prob) + str(weight_decay) + \
                str(propagation_weights).replace(' ', '') + '.png'
    plt.savefig(fname=file_name, format='png')

    fig, ax = plt.subplots(figsize=(8, 5))
    for k in range(K):
        x_len = len(avg_results['accuracy'][k])
        ax.plot(np.arange(1, x_len+1), avg_results['accuracy'][k],
                label=f'k = {k}')
    ax.set_xlabel('epochs')
    ax.set_ylabel('accuracy')
    ax.set_title('Test Accuracy')
    ax.legend()
    file_name = 'acc/' + data_set + '_' + model_type + '_' + version + \
                str(lr_rate) + str(alpha) + str(drop_prob) + str(weight_decay) + \
                str(propagation_weights).replace(' ', '') + '.png'
    plt.savefig(fname=file_name, format='png')