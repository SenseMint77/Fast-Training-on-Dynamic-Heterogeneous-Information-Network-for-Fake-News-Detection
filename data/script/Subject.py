import itertools
from typing import List


class Subject:
    id_iter = itertools.count()

    def __init__(self, subject_name: str):
        self._id = next(Subject.id_iter)
        self._subject = subject_name

    def __repr__(self) -> str:
        return f"Subject: id = {self._id}, name = {self._subject}"

    def get_id(self) -> int:
        return self._id

    def get_subject(self) -> str:
        return self._subject

    @staticmethod
    def get_subjects(subject_dict: dict) -> List[str]:
        subjects = []
        for _, item in subject_dict.items():
            subjects.append(item.get_subject())
        return subjects
