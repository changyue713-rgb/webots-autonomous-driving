"""Semantic segmentation using the trained Webots U-Net model."""

from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F

from ..types import SegmentationResult


class SemanticSegmenter:
    """
    Semantic segmentation using the trained Webots U-Net model.

    Classes:
        0: background
        1: road
        2: sky
        3: white object
        4: grey object
        5: blue object
        6: vegetation / coloured object
    """

    CLASS_NAMES = [
        "background",
        "road",
        "sky",
        "white_object",
        "grey_object",
        "blue_object",
        "vegetation_or_coloured_object",
    ]

    NUM_CLASSES = 7

    # Height x width used during training.
    IMAGE_SIZE = (64, 128)

    def __init__(self, model_path=None, device=None):
        """
        Initialise the semantic segmentation model.

        Parameters
        ----------
        model_path : str, optional
            Path to the trained model checkpoint.
            If omitted, the project's default checkpoint is used.

        device : str, optional
            Torch device such as "cpu" or "cuda".
            If omitted, CUDA is used when available.
        """

        # Select device.
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        # Create the same U-Net architecture used during training.
        self.model = smp.Unet(
            encoder_name="resnet18",
            encoder_weights=None,
            in_channels=3,
            classes=self.NUM_CLASSES,
        )

        # Locate the trained checkpoint.
        if model_path is None:
            project_root = Path(__file__).resolve().parents[3]
            model_path = (
                project_root
                / "models"
                / "segmentation_best_model.pth"
            )

        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Segmentation model weights not found:\n"
                f"{self.model_path}"
            )

        # Load trained weights.
        checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.to(self.device)
        self.model.eval()

    def segment(self, image) -> SegmentationResult:
        """
        Perform semantic segmentation on an RGB camera image.

        Parameters
        ----------
        image : numpy.ndarray
            RGB image with shape (H, W, 3).

        Returns
        -------
        SegmentationResult
            Contains a 2-D array of predicted class IDs.
        """

        if image is None:
            raise ValueError("Input image cannot be None")

        if not isinstance(image, np.ndarray):
            raise TypeError(
                "Input image must be a numpy.ndarray"
            )

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                "Input image must have shape (H, W, 3)"
            )

        original_height, original_width = image.shape[:2]

        # Convert RGB image to float32 in the range [0, 1].
        image = image.astype(np.float32) / 255.0

        # ImageNet normalisation used during training.
        mean = np.array(
            [0.485, 0.456, 0.406],
            dtype=np.float32,
        )

        std = np.array(
            [0.229, 0.224, 0.225],
            dtype=np.float32,
        )

        image = (image - mean) / std

        # H x W x C -> C x H x W.
        tensor = torch.from_numpy(image)
        tensor = tensor.permute(2, 0, 1)
        tensor = tensor.unsqueeze(0)

        # Resize to the training resolution.
        tensor = F.interpolate(
            tensor,
            size=self.IMAGE_SIZE,
            mode="bilinear",
            align_corners=False,
        )

        tensor = tensor.to(self.device)

        # Model inference.
        with torch.no_grad():
            output = self.model(tensor)

        # Select the most likely class for each pixel.
        segmentation_map = torch.argmax(
            output,
            dim=1,
        )[0]

        segmentation_map = (
            segmentation_map
            .cpu()
            .numpy()
        )

        # Resize class labels back to the original camera resolution.
        # Nearest-neighbour interpolation is required because the
        # segmentation values are discrete class IDs.
        segmentation_tensor = torch.from_numpy(
            segmentation_map.astype(np.float32)
        )

        segmentation_tensor = F.interpolate(
            segmentation_tensor.unsqueeze(0).unsqueeze(0),
            size=(original_height, original_width),
            mode="nearest",
        )

        segmentation_map = (
            segmentation_tensor[0, 0]
            .numpy()
            .astype(np.uint8)
        )

        return SegmentationResult(
            segmentation_map=segmentation_map
        )