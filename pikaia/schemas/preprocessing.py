from enum import Enum


class FeatureType(str, Enum):
    """
    Enum representing types of features in the dataset.

    Members:
        COST: A cost feature — lower values are more desirable.
        GAIN: A gain feature — higher values are more desirable.
    """

    COST = "COST"
    GAIN = "GAIN"
