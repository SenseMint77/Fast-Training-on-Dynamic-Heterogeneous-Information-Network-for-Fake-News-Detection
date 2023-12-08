import logging
import itertools
import re
from typing import Any
import numpy as np

from Author import Author
from Subject import Subject
from Label import get_label
from Source import Source

class News:
    id_iter = itertools.count()

    def __init__(self, path: str = None, ori_id: Any = None, content: str = None,
                 news_label: Any = None, author: Author = None,
                 subjects: list = [], source: Source = None):
        self.logger = logging.getLogger(__name__)

        self._path = path
        self._id = next(News.id_iter)
        # set news id
        if ori_id is None:
            self._ori_id = self._id
        else:
            self._ori_id = ori_id

        # set news content
        if content is None:
            raise ValueError("News content cannot be of NoneType.")
        self._content = self.clean_text(content)

        # set news label
        if news_label is None:
            raise ValueError("News label cannot be of NoneType.")
        self._label = get_label(news_label)

        # set author
        if author is None:
            self.logger.info("Proceeding without author...")
        self._author = author

        # set subject(s)
        self._subjects = None
        if len(subjects) == 0:
            self.logger.info("Proceeding without subjects...")
        elif not isinstance(subjects[0], Subject):
            self.logger.info("Invalid type for subjects")
        else:
            self._subjects = subjects

        # (optional) set source
        self._source = source

    def __repr__(self) -> str:
        return f"News: id = {self._id}, ori_id = {self._ori_id}, content = {self._content[:100]}, "\
            f"label = {self._label}, author = [{str(self._author)}], "\
            f"subjects = {str(self._subjects)}"

    def get_id(self) -> int:
        return self._id

    def get_path(self) -> str:
        return self._path

    def get_content(self) -> str:
        return self._content

    def get_label(self) -> int:
        return self._label

    def get_author(self) -> Author:
        return self._author

    def get_subjects(self) -> Subject:
        return self._subjects

    def get_source(self) -> Source:
        return self._source

    @ staticmethod
    def clean_text(text: str) -> str:
        text = re.sub(r'\"|ʺ', r'', text)
        text = re.sub(r'(\n)', r' ', text)
        return text

    @ staticmethod
    def get_news_infos(news_dict: dict) -> np.ndarray:
        contents = []
        labels = []
        paths = []
        for item in news_dict.values():
            if isinstance(item, News):
                contents.append(item.get_content())
                labels.append(item.get_label())
                paths.append(item.get_path())
        return np.array(contents), np.array(labels), paths

    @ staticmethod
    def get_news_author_mapping(news_dict: dict) -> dict:
        mapping = []
        for item in news_dict.values():
            if isinstance(item, News):
                mapping.append([item.get_id(), item.get_author().get_id()])
        return np.array(mapping)

    @ staticmethod
    def get_news_subject_mapping(news_dict: dict) -> dict:
        mapping = []
        for item in news_dict.values():
            if isinstance(item, News):
                subjects = item.get_subjects()
                if subjects is None:
                    continue
                if len(subjects) > 0:
                    mapping += [[item.get_id(), subject.get_id()]
                                for subject in subjects]
        return np.array(mapping)

    @ staticmethod
    def get_source_news_mapping(news_dict: dict) -> dict:
        mapping = []
        for item in news_dict.values():
            if isinstance(item, News) and item.get_source() is not None:
                mapping.append([item.get_source().get_id(), item.get_id()])
        mapping.sort()
        return np.array(mapping)
