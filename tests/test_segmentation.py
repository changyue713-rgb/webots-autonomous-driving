import numpy as np
import pytest

from elec5308_autonomy.perception.segmentation import SemanticSegmenter
from elec5308_autonomy.types import SegmentationResult


@pytest.fixture(scope="module")
def segmenter():
    """Create one segmentation model for all tests."""
    return SemanticSegmenter(device="cpu")


def test_returns_segmentation_result(segmenter):
    """Segmentation returns the expected result type and NumPy array."""
    image = np.random.randint(
        0,
        256,
        size=(480, 640, 3),
        dtype=np.uint8,
    )

    result = segmenter.segment(image)

    assert isinstance(result, SegmentationResult)
    assert isinstance(result.segmentation_map, np.ndarray)


def test_output_shape_matches_input(segmenter):
    """Segmentation map has the same height and width as the input."""
    image = np.random.randint(
        0,
        256,
        size=(480, 640, 3),
        dtype=np.uint8,
    )

    result = segmenter.segment(image)

    assert result.segmentation_map.shape == (480, 640)


def test_output_contains_valid_class_ids(segmenter):
    """All predicted class IDs are within the seven defined classes."""
    image = np.random.randint(
        0,
        256,
        size=(480, 640, 3),
        dtype=np.uint8,
    )

    result = segmenter.segment(image)

    unique_classes = np.unique(result.segmentation_map)

    assert np.all(unique_classes >= 0)
    assert np.all(unique_classes < SemanticSegmenter.NUM_CLASSES)


def test_different_image_size(segmenter):
    """Segmentation works with a different camera image resolution."""
    image = np.random.randint(
        0,
        256,
        size=(240, 320, 3),
        dtype=np.uint8,
    )

    result = segmenter.segment(image)

    assert result.segmentation_map.shape == (240, 320)


def test_non_numpy_input_raises_type_error(segmenter):
    """Non-NumPy input is rejected."""
    image = [[0, 0, 0]]

    with pytest.raises(TypeError):
        segmenter.segment(image)


def test_invalid_image_shape_raises_value_error(segmenter):
    """Images that do not have shape (H, W, 3) are rejected."""
    image = np.random.randint(
        0,
        256,
        size=(480, 640),
        dtype=np.uint8,
    )

    with pytest.raises(ValueError):
        segmenter.segment(image)