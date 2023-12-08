import numpy as np
import random
import torch
import re

from torch.utils.data import DataLoader, Dataset
from transformers import BertModel
from transformers import AdamW, get_scheduler, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score

MAX_SEQ_LENGTH = 250
NUM_OVERLAP = 50


class FakeNewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item


class FineTunedBert(torch.nn.Module):
    def __init__(self):
        super(FineTunedBert, self).__init__()
        self.bert = BertModel.from_pretrained(
            'bert-base-uncased', output_hidden_states=True)
        self.lin = torch.nn.Linear(768, 2)

    def forward(self, input_ids, token_type_ids, attention_mask):
        output = self.bert(input_ids, token_type_ids, attention_mask)
        hidden_states = torch.stack(output[2], dim=0)
        pooled_output = torch.sum(hidden_states[-4:], dim=0)
        out = torch.mean(pooled_output, dim=1)
        out = self.lin(out)
        return out


class FineTunedBertTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        model_input = {k: v for k, v in inputs.items() if k != 'labels'}
        logits = model(**model_input)
        loss_function = torch.nn.CrossEntropyLoss()
        loss = loss_function(logits, inputs['labels'])
        return (loss, (loss, logits)) if return_outputs else loss


def get_data_group(split_tokenized_sequence, split_labels, idx, mask):
    data = {'labels': split_labels[mask], 'idx': idx[mask]}
    data['encodings'] = {'input_ids': [],
                         'token_type_ids': [], 'attention_mask': []}
    for idx in mask:
        data['encodings']['input_ids'].append(
            split_tokenized_sequence['input_ids'][idx])
        data['encodings']['token_type_ids'].append(
            split_tokenized_sequence['token_type_ids'][idx])
        data['encodings']['attention_mask'].append(
            split_tokenized_sequence['attention_mask'][idx])
    return data


def get_split_labels(split_idx, labels):
    split_labels = []
    for idx in split_idx:
        split_labels.append(labels[idx])
    return np.array(split_labels)


def split_data(split_tokenized_corpus, split_idx, labels, ratio, seed=1234):
    data = {'train': {}, 'val': {}}
    n_train = round(ratio[0]*len(split_idx))
    random_indices = np.arange(len(split_idx))
    random.Random(seed).shuffle(random_indices)
    train_mask = random_indices[:n_train]
    val_mask = random_indices[n_train:len(split_idx)]
    split_labels = get_split_labels(split_idx, labels)
    data['train'] = get_data_group(
        split_tokenized_corpus, split_labels, split_idx, train_mask)
    data['val'] = get_data_group(
        split_tokenized_corpus, split_labels, split_idx, val_mask)
    return data


def compute_metrics(eval_pred):
    preds, true_labels = eval_pred
    pred_labels = np.argmax(preds, axis=1)
    accuracy = accuracy_score(true_labels, pred_labels)
    return {'accuracy': accuracy}


def bert_fine_tune(logger, data_set, split_tokenized_corpus, split_idx, labels):
    # parameters
    batch_size = 16
    n_epochs = 10
    ratio = [0.7, 0.3]

    # load data into training and validation sets
    data = split_data(split_tokenized_corpus,
                      np.array(split_idx), labels, ratio)
    train_dataset = FakeNewsDataset(
        data['train']['encodings'], data['train']['labels'])
    val_dataset = FakeNewsDataset(
        data['val']['encodings'], data['val']['labels'])

    # initialize model
    model = FineTunedBert()
    # train model and choose the best weights
    training_args = TrainingArguments(
        data_set + "_bert_trainer", learning_rate=0.001, num_train_epochs=n_epochs, evaluation_strategy="epoch",
        logging_strategy="steps", logging_steps=100, save_strategy="epoch", save_total_limit=100,
        seed=1234, per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
        load_best_model_at_end=True, metric_for_best_model='accuracy', dataloader_pin_memory=False)
    trainer = FineTunedBertTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    trainer.train()

    # return BERT model weights
    return {re.sub(r'^bert.', '', k): v for k, v in model.state_dict().items()
            if re.match(r'^bert', k)}
