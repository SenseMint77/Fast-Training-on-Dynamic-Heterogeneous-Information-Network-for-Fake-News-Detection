import itertools
from typing import List


class Source:
    id_iter = itertools.count()

    def __init__(self, source: str):
        self._id = next(Source.id_iter)
        self._source = source

    def __repr__(self) -> str:
        return f"Source: id = {self._id}, name = {self._source}"

    def get_id(self) -> int:
        return self._id

    def get_source(self) -> str:
        return self._source

    @staticmethod
    def get_sources(source_dict: dict) -> List[str]:
        sources = []
        for _, item in source_dict.items():
            sources.append(item.get_source())
        return sources
