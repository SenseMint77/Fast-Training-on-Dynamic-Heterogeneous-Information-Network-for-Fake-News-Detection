import torch
import re
import numpy as np
from transformers import BertTokenizer, BertModel
from bert_finetuning import bert_fine_tune


class SmallerBert(torch.nn.Module):
    def __init__(self):
        super(SmallerBert, self).__init__()
        self.bert = BertModel.from_pretrained(
            'bert-base-uncased', output_hidden_states=True)
        self.lin = torch.nn.Linear(768, 64)

    def load_state_dict(self, bert_weight):
        self.bert.load_state_dict(bert_weight)

    def forward(self, input_ids, token_type_ids, attention_mask):
        h = self.bert(input_ids, token_type_ids, attention_mask)
        h = torch.stack(h[2], dim=0)
        out = self.lin(h)
        return out


def get_bert_embed_vector(data_set, corpus, labels, logger, with_bert_finetuning):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # use pretrained uncased base bert model tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenized_corpus, empty_corpus = get_tokenized_input(tokenizer, corpus)
    # get token-index mapping
    token_indices = save_token_index(
        tokenizer, tokenized_corpus['input_ids'])
    # split long token encodings
    max_length = 510
    n_overlap = 50
    start_token = torch.tensor([empty_corpus['input_ids'][0][0]])
    end_token = torch.tensor([empty_corpus['input_ids'][0][1]])
    split_tokenized_corpus, split_idx = split_encodings(
        tokenized_corpus, start_token, end_token, max_length, n_overlap)
    split_tokenized_corpus = {key: torch.stack((item)).to(torch.int).to(device)
                              for key, item in split_tokenized_corpus.items()}

    # Load pre-trained or finetuned model
    bert_model = SmallerBert()
    # use cuda if available
    if with_bert_finetuning:
        bert_model_weight = bert_fine_tune(
            logger, data_set, split_tokenized_corpus, split_idx, labels)
        bert_model.load_state_dict(bert_model_weight)
    if torch.cuda.device_count() > 1:
        print(f"--- Using {torch.cuda.device_count()} GPUs ---")
        bert_model = torch.nn.DataParallel(bert_model)
    bert_model.eval()

    # get embedding vector
    torch.cuda.empty_cache()
    split_embed_corpus = evaluate_corpus(
        bert_model.to(device), split_tokenized_corpus)

    # aggregate results from split
    embed_corpus = aggregate_embeddings(
        split_embed_corpus, split_idx, n_overlap)

    return embed_corpus, token_indices


def split_encodings(encodings, start_token, end_token, max_length=510, n_overlap=50):
    if n_overlap > max_length:
        raise ValueError('max_len has to be grater than n_overlap')
    net_length = max_length - n_overlap
    idx = []
    splits = {key: [] for key in encodings.keys()}
    for i, _ in enumerate(encodings['input_ids']):
        encoding_length = len(encodings['input_ids'][i])
        j = 0
        while(True):
            idx.append(i)
            start_id = j*net_length
            if start_id + max_length < encoding_length:
                splits = add_encoding(
                    encodings, splits, i, start_id, start_id+max_length, max_length+2,
                    start_token, end_token)
                j += 1
            else:
                splits = add_encoding(
                    encodings, splits, i, start_id, encoding_length, max_length+2,
                    start_token, end_token)
                break
    return splits, idx


def add_encoding(encodings, splits, idx, start_id, end_id, max_length,
                 start_token, end_token):
    for key in encodings.keys():
        buf = encodings[key][idx][start_id:end_id]
        if key == 'input_ids':
            # add [CLS] and [SEP] to start and end
            buf = torch.cat((start_token, buf, end_token))
        elif key == 'attention_mask':
            buf = torch.cat((buf, torch.tensor([1, 1])))
        if len(buf) < max_length:
            buf = torch.cat((buf, torch.zeros(max_length - len(buf))))
        splits[key].append(buf)
    return splits


def aggregate_embeddings(embeddings, idx, n_overlap):
    result = []
    buf = []
    for i, embedding in enumerate(embeddings):
        if i == 0:
            buf = embedding
        elif idx[i] == idx[i-1]:
            buf = np.vstack((buf[:-1], embedding[n_overlap+1:]))
        else:
            result.append(buf)
            buf = embedding
    result.append(buf)
    return result


def get_tokenized_input(tokenizer, corpus):
    tokenized_corpus = {'input_ids': [],
                        'token_type_ids': [],
                        'attention_mask': []}
    for text in corpus:
        # text = re.sub(r'(\'|\’)', r'', text)
        # text = re.sub(r'\W+', r' ', text)
        tokens = tokenizer.encode_plus(
            text, add_special_tokens=False, return_tensors='pt')
        for key in tokenized_corpus.keys():
            tokenized_corpus[key].append(tokens[key][0])
    empty_corpus = tokenizer('', return_tensors='pt')
    return tokenized_corpus, empty_corpus


def evaluate_corpus(model, tokenized_corpus):
    corpus_embed = []
    with torch.no_grad():
        len_corpus = len(tokenized_corpus['input_ids'])
        for i in range(len_corpus):
            output = model(input_ids=tokenized_corpus['input_ids'][i:i+1],
                           token_type_ids=tokenized_corpus['token_type_ids'][i:i+1],
                           attention_mask=tokenized_corpus['attention_mask'][i:i+1])
            # get word embedding vector as the sum of the last four hidden layers
            token_embed = torch.squeeze(output, dim=1)
            word_embed = torch.sum(
                token_embed[-4:], dim=0).to(dtype=torch.float16)
            corpus_embed.append(word_embed.cpu().numpy())
    return corpus_embed


def get_vocab(tokenizer, token_ids):
    vocab = {}
    for idx, _ in enumerate(token_ids):
        text = tokenizer.convert_ids_to_tokens(token_ids[idx])
        for tup in zip(text, token_ids[idx]):
            if tup[0] == '[PAD]':
                vocab[tup[0]] = int(tup[1])
                break
            vocab[tup[0]] = int(tup[1])
    return vocab


def save_token_index(tokenizer, token_ids):
    token_indices = []
    for idx, _ in enumerate(token_ids):
        text = tokenizer.convert_ids_to_tokens(token_ids[idx])
        token_indices.append(list(zip(text, token_ids[idx].numpy())))
    return token_indices
