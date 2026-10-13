import numpy as np
import pytest
from PIL import Image

from elec5308_autonomy.perception.object_detection import ObjectDetector
from elec5308_autonomy.types import ObjectDetectionResult


def test_detect_returns_correct_result_type():
    """detect() should return an ObjectDetectionResult with the right fields."""
    detector = ObjectDetector()
    fake_image = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype='uint8'))

    result = detector.detect(fake_image)

    assert isinstance(result, ObjectDetectionResult)
    assert isinstance(result.objects, list)
    assert isinstance(result.bounding_boxes, list)
    assert isinstance(result.confidence, list)
    assert len(result.objects) == len(result.bounding_boxes) == len(result.confidence)
def test_detect_bounding_boxes_have_four_coordinates():
    """Each bounding box should be a list of exactly 4 coordinates (x1, y1, x2, y2)."""
    detector = ObjectDetector()
    fake_image = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype='uint8'))

    result = detector.detect(fake_image)

    for box in result.bounding_boxes:
        assert len(box) == 4
        assert all(isinstance(coord, float) for coord in box)


def test_detect_confidence_scores_are_valid_probabilities():
    """Confidence scores should all be floats between 0 and 1."""
    detector = ObjectDetector()
    fake_image = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype='uint8'))

    result = detector.detect(fake_image)

    for score in result.confidence:
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


def test_detect_handles_different_image_sizes():
    """detect() should not crash on a much smaller image size."""
    detector = ObjectDetector()
    small_image = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype='uint8'))

    result = detector.detect(small_image)

    assert isinstance(result, ObjectDetectionResult)

def test_detect_raises_typeerror_on_invalid_input():
    """detect() should raise a clear TypeError if given something other than a PIL Image."""
    detector = ObjectDetector()

    with pytest.raises(TypeError):
        detector.detect("not an image")