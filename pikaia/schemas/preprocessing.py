from enum import Enum


class FeatureType(str, Enum):
    """
    Enum representing types of features in the dataset.

    Attributes:
        COST (str): Represents a cost feature. Higher values are less desirable.
        GAIN (str): Represents a gain feature. Higher values are more desirable.
    """

    COST = "COST"
    GAIN = "GAIN"
