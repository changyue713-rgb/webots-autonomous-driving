import torch
from PIL import Image
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.transforms.functional import to_tensor

from ..types import ObjectDetectionResult


class ObjectDetector:
    """Detects and localises objects in a scene using a pretrained Faster R-CNN model.

    Uses torchvision's fasterrcnn_resnet50_fpn with COCO-pretrained weights.
    Only detections above `confidence_threshold` are returned, to filter out
    low-confidence noise.
    """

    def __init__(self, confidence_threshold: float = 0.5):
        """Load the pretrained detection model.

        Args:
            confidence_threshold: Minimum confidence score (0-1) for a detection
                to be included in the result. Defaults to 0.5.
        """
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT

        # Run on the GPU when one is available. Faster R-CNN is far too slow on
        # CPU to keep up with a live simulation loop.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = fasterrcnn_resnet50_fpn(weights=weights).to(self.device)
        self.model.eval()
        self.categories = weights.meta["categories"]
        self.confidence_threshold = confidence_threshold

    def detect(self, image: Image.Image) -> ObjectDetectionResult:
        """Detect and localise objects from an input image.

        Args:
            image: A PIL Image to run detection on.

        Returns:
            An ObjectDetectionResult containing the detected object labels,
            bounding boxes, and confidence scores, filtered to only include
            detections above confidence_threshold.

        Raises:
            TypeError: If `image` is not a PIL Image.
        """
        if not isinstance(image, Image.Image):
            raise TypeError(
                f"ObjectDetector.detect() expects a PIL Image, got {type(image).__name__}"
            )

        image_tensor = to_tensor(image).to(self.device)

        with torch.no_grad():
            predictions = self.model([image_tensor])[0]

        labels = []
        boxes = []
        scores = []

        for label_idx, box, score in zip(
            predictions["labels"].tolist(),
            predictions["boxes"].tolist(),
            predictions["scores"].tolist(),
        ):
            if score >= self.confidence_threshold:
                labels.append(self.categories[label_idx])
                boxes.append(box)
                scores.append(score)

        return ObjectDetectionResult(
            objects=labels,
            bounding_boxes=boxes,
            confidence=scores,
        )