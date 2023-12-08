import logging, getopt, sys
import numpy as np
from pathlib import Path

from load_data import load_dataset
from save_data import save_data
from bert_embedding import get_bert_embed_vector
from graph_utils import get_feature_matrices, get_adjacency_matrices
from Author import Author
from News import News

WINDOW_SIZE = 3
data_dir = Path(__file__).parent.parent


def get_args(argv):
    try:
        opts, _ = getopt.getopt(argv,"hd:v:",["data_set=","version="])
    except getopt.GetoptError:
        print ('test.py -d <data_set> -v <version>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == 'h':
            print ('test.py -d <data_set> -v <version>')
            sys.exit()
        elif opt in ("-d", "--data_set"): # liar_dataset or FakeNewsNet
            data_set = arg
        elif opt in ("-v", "--version"):  # no_finetuning or with_finetuning
            version = arg
    return data_set, version

if __name__ == "__main__":
    data_set, version = get_args(sys.argv[1:])
    with_bert_finetuning = True if version == 'with_finetuning' else False

    log_filename = 'log/' + data_set + '_' + version + '_log.out'
    logging.basicConfig(filename=log_filename, filemode='w', 
                        format='%(asctime)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)

    # load data
    news_dict, author_dict, subjects, sources = load_dataset(logger, data_set)
    # get news and and author information
    contents, news_labels, paths = News.get_news_infos(news_dict)
    profiles, author_labels = Author.get_author_infos(author_dict)
    news_author_mapping = News.get_news_author_mapping(news_dict)
    # get source and subject mappings
    news_subject_mapping = News.get_news_subject_mapping(news_dict)
    author_source_mapping = Author.get_author_source_mapping(author_dict)

    # print(f'#news: {len(contents)}, #author: {len(profiles)},')
    # print(f'#subject: {len(subjects)}, #source: {len(sources)}')

    # use bert to get embedding vectors and token-index mapping
    embed_contents, contents_mapping = get_bert_embed_vector(
        data_set, contents, news_labels, logger, with_bert_finetuning)
    embed_profiles, profiles_mapping = get_bert_embed_vector(
        data_set, profiles, author_labels, logger, with_bert_finetuning)
    # get feature matrices
    contents_feature_matrices = get_feature_matrices(
        contents, embed_contents, contents_mapping)
    profiles_feature_matrices = get_feature_matrices(
        profiles, embed_profiles, profiles_mapping)

    # get adjacency matrices
    contents_adjacency_matrices = get_adjacency_matrices(
        contents, contents_feature_matrices, WINDOW_SIZE)
    profiles_adjacency_matrices = get_adjacency_matrices(
        profiles, profiles_feature_matrices, WINDOW_SIZE)

    # save graphs
    save_data('news', data_set, version, paths, contents_adjacency_matrices,
              contents_feature_matrices, news_labels)
    save_data('author', data_set, version, paths, profiles_adjacency_matrices,
              profiles_feature_matrices, author_labels)
    path = str(paths[0])
    root_dir = Path(path[:path.find(data_set) + len(data_set)])
    with open(root_dir / 'news_author.npz', 'wb') as f:
        np.savez(f, mapping=news_author_mapping)
    with open(root_dir / 'news_subject.npz', 'wb') as f:
        np.savez(f, mapping=news_subject_mapping)
    with open(root_dir / 'author_source.npz', 'wb') as f:
        np.savez(f, mapping=author_source_mapping)
    if data_set == 'FakeNewsNet':
        with open(root_dir / 'paths.npz', 'wb') as f:
            np.savez(f, paths=paths)

    # log stats
    logger.info("contents --> size: %s", contents.shape)
    logger.info("             adj_matrices: %s",
                len(contents_adjacency_matrices))
    logger.info("             feature_matrices: %s",
                len(contents_feature_matrices))
    logger.info("             labels: %s", news_labels.shape)

    logger.info("profiles --> size: %s", profiles.shape)
    logger.info("             adj_matrices: %s",
                len(profiles_adjacency_matrices))
    logger.info("             feature_matrices: %s",
                len(profiles_feature_matrices))
    logger.info("             labels: %s", author_labels.shape)
