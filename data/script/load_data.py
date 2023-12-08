import csv
import os
import json
import re
import numpy as np
from pathlib import Path
from itertools import product

from News import News
from Author import Author
from Subject import Subject
from Source import Source

data_dir = Path(__file__).parent.parent
MAX_TEXT_LENGTH = 1000


def load_dataset(logger, data_set):
    path_to_file = data_dir / data_set
    if data_set == 'liar_dataset':
        return load_liar(logger, path_to_file)
    elif data_set == 'FakeNewsNet':
        return load_fnn(logger, path_to_file)


def load_liar(logger, root_dir):
    file_names = ['train.tsv', 'test.tsv', 'valid.tsv']
    clean_file_names = ['train_clean.tsv',
                        'test_clean.tsv', 'valid_clean.tsv']

    for i, _ in enumerate(file_names):
        with open(root_dir / file_names[i], 'r', encoding='utf-8') as handler:
            text = handler.read()
        text = re.sub(r'\"|ʺ', r'', text)
        with open(root_dir / clean_file_names[i], 'w', encoding='utf-8') as handler:
            handler.write(text)

    news_articles = {}
    authors = {}
    subjects = {}
    sources = {}
    for file_name in clean_file_names:
        with open(root_dir / file_name, 'r', encoding='utf-8') as handler:
            logger.info("extracting data from %s...", file_name)
            read_tsv = csv.reader(handler, delimiter='\t', quotechar='"')
            for row in read_tsv:
                # get source info
                source_name = row[13]
                source = None
                if source_name in sources:
                    source = sources[source_name]
                else:
                    source = Source(source_name)
                    sources[source_name] = source
                # get author object
                author_name = row[4].replace('-', ' ')
                profile = ' '.join(row[5:8])
                author_label = Author.get_author_label(np.array(row[8:13]))
                if author_name in authors:
                    author = authors[author_name]
                    if source not in author.get_sources():
                        author.add_source(source)
                else:
                    author = Author(None, author_label,
                                    author_name, profile, source)
                    authors[author_name] = author
                # get subject object(s)
                subject_names = row[3].split(',')
                news_subjects = []
                for subject_name in subject_names:
                    if subject_name in subjects:
                        news_subjects.append(subjects[subject_name])
                    else:
                        subject = Subject(subject_name)
                        news_subjects.append(subject)
                        subjects[subject_name] = subject
                # create news object
                news_id = row[0].replace('.json', '')
                label = row[1]
                content = row[2]
                news = News(root_dir, news_id,
                            content, label, author, news_subjects, source)
                # add news to dictionaries
                news_articles[news.get_id()] = news
                logger.info("processed %s", str(news))
            logger.info("news article size: %d, authors size: %d, "
                        "subjects size: %d, sources size: %d",
                        len(news_articles), len(authors), len(subjects), len(sources))
    return news_articles, authors, subjects, sources


def load_fnn(logger, root_dir):
    platforms = ["gossipcop", "politifact"]
    labels = ["fake", "real"]
    news_articles = {}
    unknown_author = Author(None, None, 'unknown', 'unknown')
    authors = {'unknown': unknown_author}
    sources = {}
    subjects = {}
    for platform, label in product(platforms, labels):
        # recursively traverse the directories
        root = root_dir / platform / label
        path = os.walk(root)
        for _, directories, _ in path:
            for directory in directories:
                if not re.match(r'gossipcop-|politifact', directory):
                    continue
                parent_dir = root / directory
                short_dir = platform + '/' + label + '/' + directory
                files = os.listdir(path=parent_dir)
                if len(files) == 0:
                    logger.info('Skipping %s: empty folder', short_dir)
                    continue
                filename = parent_dir / 'news content.json'
                # load news content from json file
                if not os.path.exists(filename):
                    logger.info('Skipping %s: no news content', short_dir)
                    continue
                with open(filename, encoding='utf-8') as loader:
                    news_dict = json.load(loader)
                if news_dict['text'] == "":
                    logger.info('Skipping %s: no text', short_dir)
                    continue
                elif len(news_dict['text'].split(" ")) > MAX_TEXT_LENGTH:
                    logger.info('Skipping %s: text is too long', short_dir)
                    continue
                # get source if exists
                source = None
                if news_dict.get('meta_data') and news_dict['meta_data'].get('og')\
                        and news_dict['meta_data']['og'].get('site_name'):
                    source_name = news_dict['meta_data']['og']['site_name']
                    if source_name in sources:
                        source = sources[source_name]
                    else:
                        source = Source(source_name)
                        sources[source_name] = source
                # get author
                if len(news_dict['authors']) == 0:
                    if source is not None:
                        author_name = source_name
                    else:
                        author_name = 'unknown'
                else:
                    author_name = news_dict['authors'][0]
                if author_name in authors:
                    author = authors[author_name]
                    author.add_news_label(label)
                else:
                    author = Author(None, None, author_name,
                                    author_name, source, [label])
                authors[author_name] = author
                # get subjects
                subject_names = news_dict['keywords']
                news_subjects = []
                for subject_name in subject_names:
                    if subject_name in subjects:
                        news_subjects.append(subjects[subject_name])
                    else:
                        subject = Subject(subject_name)
                        news_subjects.append(subject)
                        subjects[subject_name] = subject
                # create news object
                news_id = directory
                content = news_dict['text']
                news = News(parent_dir, news_id, content,
                            label, author, news_subjects, source)
                news_articles[news.get_id()] = news
                logger.info("processed %s", str(news))
    logger.info("news article size: %d, authors size: %d",
                len(news_articles), len(authors))
    return news_articles, authors, subjects, sources
