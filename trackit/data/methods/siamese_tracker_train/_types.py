import numpy as np
from dataclasses import dataclass
from typing import Callable, Union
from trackit.data.source import TrackingDataset_Sequence, TrackingDataset_Track, TrackingDataset_FrameInTrack


@dataclass(frozen=True)
class SOTFrameInfo:
    image: Callable[[], np.ndarray]
    object_bbox: np.ndarray
    object_exists: bool
    sequence: TrackingDataset_Sequence
    track: TrackingDataset_Track
    frame: TrackingDataset_FrameInTrack


# Zekai Shao: Add MMOTFrameInfo
@dataclass(frozen=True)
class MMOTFrameInfo:
    image: Callable[[], np.ndarray]
    object_bbox: np.ndarray
    object_exists: bool
    sequence: TrackingDataset_Sequence
    track: TrackingDataset_Track
    frame: TrackingDataset_FrameInTrack


@dataclass(frozen=True)
class SiameseTrainingPair:
    is_positive: bool
    template: Union[SOTFrameInfo, MMOTFrameInfo]
    search: Union[SOTFrameInfo, MMOTFrameInfo]
    online: Union[SOTFrameInfo, MMOTFrameInfo]
