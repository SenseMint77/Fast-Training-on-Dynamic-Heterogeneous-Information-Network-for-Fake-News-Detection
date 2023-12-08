import itertools
import logging
from typing import Any
import numpy as np

from Source import Source
from Label import get_label


class Author:
    id_iter = itertools.count()

    def __init__(self, ori_id: Any = None, author_label: Any = None,
                 author_name: str = None, profile: str = None,
                 source: Source = None, news_labels: list = []):
        self.logger = logging.getLogger(__name__)

        self._id = next(Author.id_iter)
        # set author id
        if ori_id is None:
            self._ori_id = self._id
        else:
            self._ori_id = ori_id

        # set author name and profile
        if author_name is None:
            raise ValueError('Author name cannot be of NoneType.')
        self._name = self.clean_text(author_name)
        self._profile = self.clean_text(profile)

        # set author label or alternatively news labels
        self._label = None
        if author_label is not None:
            self._label = get_label(author_label)
        if len(news_labels) == 0:
            self._news_labels = []
        else:
            self._news_labels = [get_label(label) for label in news_labels]

        # set source
        self._sources = []
        if not isinstance(source, Source):
            self.logger.info("Proceeding without source...")
        else:
            self._sources = [source]

    def __repr__(self) -> str:
        return f"Author: id = {self._id}, name = {self._name}, "\
            f"profile = {self._profile}, label = {self._label}, "\
            f"source = {str(self._sources)}, news_labels = {self._news_labels}"

    def get_id(self) -> Any:
        return self._id

    def get_name(self) -> str:
        return self._name

    def get_profile(self) -> Any:
        return self._profile

    def get_sources(self) -> Source:
        return self._sources

    def get_label(self) -> int:
        if self._label is None:
            self._label = round(np.mean(self._news_labels))
        return self._label

    def add_news_label(self, label: Any) -> None:
        self._news_labels.append(get_label(label))

    def add_source(self, source: Source) -> None:
        self._sources.append(source)

    @staticmethod
    def clean_text(text: str):
        # (optional) remove some emojis from text
        # emoji_pattern = re.compile("["
        #                            u"\U0001F600-\U0001F64F"  # emoticons
        #                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        #                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
        #                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        #                            "]+", flags=re.UNICODE)
        # text = emoji_pattern.sub(r'', text)
        return text

    @staticmethod
    def get_author_label(score_arr: list):
        """
        Average Score:   1 * [#Pants on Fire] + 2 * [#False] + 3 * [#Barely True]
                       + 4 * [#Half True]     + 5 * [#Mostly True]
        If average score < 2.5, label as 0 (false)
        Else, label as 1 (true)
        Look at LIAR data set's README file for more details on the scores
        """
        if len(score_arr) == 0:
            raise ValueError('scores cannot be empty: %s', score_arr)
        weights = [3, 2, 4, 5, 1]
        avg_score = np.mean([weights[i] * int(score_arr[i])
                            for i in range(len(score_arr))])
        if avg_score < 2.5:
            return 0
        return 1

    @staticmethod
    def get_author_infos(author_dict: dict) -> np.ndarray:
        profiles = []
        labels = []
        for item in author_dict.values():
            if isinstance(item, Author):
                profiles.append(item.get_profile())
                labels.append(item.get_label())
        return np.array(profiles), np.array(labels)

    @ staticmethod
    def get_author_source_mapping(author_dict: dict) -> dict:
        mapping = []
        for item in author_dict.values():
            if isinstance(item, Author):
                sources = item.get_sources()
                if len(sources) > 0:
                    mapping += [[item.get_id(), source.get_id()]
                                for source in sources]
        return np.array(mapping)
