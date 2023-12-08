from typing import Any

label = {
    'true': 1,
    'half-true': 1,
    'barely-true': 1,
    'mostly-true': 1,
    'real': 1,
    'false': 0,
    'fake': 0,
    'pants-fire': 0
}


def convert_label(str_label: str):
    return label[str_label.casefold()]


def get_label(any_label: Any):
    if any_label is None:
        raise ValueError("Labels cannot be of NoneType.")
    if any_label == 0 or any_label == 1:
        return any_label
    if convert_label(any_label) is not None:
        return convert_label(any_label)
    raise ValueError(
        f"{any_label} - Labels are either 1/true/real or 0/false/fake.")
