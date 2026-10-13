import cv2
import numpy as np
from PIL import Image

from elec5308_autonomy.perception.lane_detection import LaneDetector
from elec5308_autonomy.types import LaneDetectionResult


def create_lane_image():
    """Create a simple synthetic image containing left and right lane lines."""
    image = np.zeros((64, 128, 3), dtype=np.uint8)

    # Left yellow lane
    cv2.line(
        image,
        (25, 63),
        (50, 35),
        (0, 255, 255),
        3
    )

    # Right white lane
    cv2.line(
        image,
        (103, 63),
        (78, 35),
        (255, 255, 255),
        3
    )

    return image


def test_detector_returns_lane_detection_result():
    """Detector should return the expected result type."""
    detector = LaneDetector()
    image = create_lane_image()

    result = detector.detect(image)

    assert isinstance(result, LaneDetectionResult)


def test_detect_both_lane_boundaries():
    """Detector should detect both left and right lane boundaries."""
    detector = LaneDetector()
    image = create_lane_image()

    result = detector.detect(image)

    assert len(result.left_lane) > 0
    assert len(result.right_lane) > 0


def test_driving_area_is_valid_mask():
    """Driving area should be a non-empty image-sized mask."""
    detector = LaneDetector()
    image = create_lane_image()

    result = detector.detect(image)

    assert isinstance(result.driving_area, np.ndarray)
    assert result.driving_area.shape == (64, 128)
    assert result.driving_area.dtype == np.uint8
    assert np.count_nonzero(result.driving_area) > 0


def test_detector_accepts_pil_image():
    """Detector should accept PIL Image input."""
    detector = LaneDetector()
    image = create_lane_image()

    pil_image = Image.fromarray(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    )

    result = detector.detect(pil_image)

    assert isinstance(result, LaneDetectionResult)
    assert result.driving_area.shape == (64, 128)


def test_single_lane_boundary_fallback():
    """Detector should create a driving area when only one lane is visible."""
    detector = LaneDetector()

    # First frame: establish a previous centreline.
    both_lanes = create_lane_image()
    first_result = detector.detect(both_lanes)

    assert len(first_result.left_lane) > 0
    assert len(first_result.right_lane) > 0

    # Second frame: only the right lane remains.
    image = np.zeros((64, 128, 3), dtype=np.uint8)

    cv2.line(
        image,
        (103, 63),
        (78, 35),
        (255, 255, 255),
        3
    )

    result = detector.detect(image)

    assert len(result.right_lane) > 0
    assert np.count_nonzero(result.driving_area) > 0