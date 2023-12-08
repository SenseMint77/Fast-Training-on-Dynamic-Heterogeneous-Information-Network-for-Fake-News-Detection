import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from torch import tensor
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_mean

def train(model, raw_data, optimizer, criterion, device='cpu'):
    model.train()
    optimizer.zero_grad()
    data = raw_data
    out = model(data.x.float().to(device), tensor(data.train_mask).to(device))
    loss = criterion(out, data.y[data.train_mask].to(device))
    pred = out.argmax(dim=1).cpu()
    loss.backward()   # Derive gradients
    optimizer.step()  # Update parameters based on gradients
    return loss.cpu(), accuracy_score(data.y[data.train_mask], pred)


def validate(model, raw_data, device='cpu'):
    model.eval()
    data = raw_data
    out = model(data.x.float().to(device), tensor(data.val_mask).to(device))
    pred = out.argmax(dim=1).cpu()
    val_acc = accuracy_score(data.y[data.val_mask], pred)
    return val_acc


def test(model, raw_data, device='cpu') -> dict:
    model.eval()
    data = raw_data
    out = model(data.x.float().to(device), tensor(data.test_mask).to(device))
    pred = out.argmax(dim=1).cpu()
    true_labels = data.y[data.test_mask]
    pred_labels = pred
    # Derive metrics results
    # print(confusion_matrix(true_labels, pred_labels, labels=[0,1]))
    test_results = {}
    test_results['accuracy'] = accuracy_score(true_labels, pred_labels)
    test_results['precision'] = precision_score(true_labels, pred_labels)
    test_results['recall'] = recall_score(true_labels, pred_labels)
    test_results['auc'] = roc_auc_score(true_labels, pred_labels)
    test_results['F1'] = f1_score(true_labels, pred_labels)
    return test_results


def normal_train(model, raw_data, optimizer, criterion, batch_size, device='cpu'):
    model.train()
    optimizer.zero_grad()
    loader = DataLoader(dataset=raw_data, batch_size=batch_size, shuffle=False)
    losses = []
    for data in loader:
        out = model(data.x.float().to(device), data.edge_index.to(device))
        pred = scatter_mean(out, data.batch.to(device), dim=0)
        losses.append(criterion(pred, data.y.to(device)))
    loss = torch.mean(torch.stack(losses))
    loss.backward()   # Derive gradients
    optimizer.step()  # Update parameters based on gradients
    return loss


def normal_test(model, raw_data, batch_size, device='cpu') -> dict:
    model.eval()
    loader = DataLoader(dataset=raw_data, batch_size=batch_size, shuffle=False)
    test_results = {'accuracy': [], 'precision': [],
                    'recall': [], 'auc': [], 'F1': []}
    pred_labels = []
    true_labels = []
    for data in loader:
        out = model(data.x.float().to(device), data.edge_index.to(device))
        pred = scatter_mean(out, data.batch.to(device), dim=0)
        pred = pred.argmax(dim=1).cpu()
        # Derive metrics results
        pred_labels += pred
        true_labels += data.y
    # print(confusion_matrix(true_labels, pred_labels, labels=[0,1]))
    test_results['accuracy'] = accuracy_score(true_labels, pred_labels)
    test_results['precision'] = precision_score(true_labels, pred_labels)
    test_results['recall'] = recall_score(true_labels, pred_labels)
    test_results['auc'] = roc_auc_score(true_labels, pred_labels)
    test_results['F1'] = f1_score(true_labels, pred_labels)
    return test_results