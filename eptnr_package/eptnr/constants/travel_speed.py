import enum


class MetricTravelSpeed(enum.Enum):
    WALKING = 5  # km/h
    BIKING = 15  # km/h


class ImperialTravelSpeed(enum.Enum):
    WALKING = 3.1  # mph
    BIKING = 9.32  # mph
