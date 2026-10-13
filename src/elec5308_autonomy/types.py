from dataclasses import dataclass
from typing import Any, Dict, List, Union

import numpy as np


@dataclass
class LaneDetectionResult:
    left_lane: list
    right_lane: list
    driving_area: np.ndarray


@dataclass
class ObjectDetectionResult:
    objects: Any
    bounding_boxes: Any
    confidence: Any


@dataclass
class SegmentationResult:
    segmentation_map: Any


@dataclass
class TrafficDetectionResult:  # Define the output structure for traffic detection
    traffic_signs: List[Dict[str, Union[str, float]]]  # Store information about detected traffic signs
    traffic_conditions: Dict[str, str]  # Store information about detected traffic conditions


@dataclass
class NavigationInfo:
    path: Any
    target_point: Any


@dataclass
class SteeringCommand:
    steering_angle: Any


@dataclass
class SpeedCommand:
    speed: Any