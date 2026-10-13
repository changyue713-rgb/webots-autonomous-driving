# from ..types import TrafficDetectionResult  # Import the output data structure for traffic detection
# import torch
# import torchvision.transforms as transforms
# from PIL import Image

# class TrafficDetector:  # Define a class for traffic condition detection
#     """Interface for detecting traffic conditions and signs."""  # Describe the purpose of this class

#     def __init__(self, model_name: str = "resnet18"):
#         """
#         Initializes the traffic detector.
#         :param model_name: "resnet18"
#         """
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model_name = model_name.lower()
        
#         # Load the corresponding model architecture dynamically
#         self.model = self._load_architecture(self.model_name)

#         from pathlib import Path
#         weights_path = (
#           Path(__file__).parent.parent.parent.parent / "models" / "best_model.pth"
#       ) 
#         if weights_path.exists():
#           self.model.load_state_dict(
#               torch.load(weights_path, map_location=self.device)
#         )

#         self.model.eval()
#         self.model.to(self.device)
        
#         # Image preprocessing
#         self.transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])

#     def _load_architecture(self, name: str):
#         """Helper to load and adapt the requested model architecture for 43 GTSRB classes."""
#         if "resnet" in name:
#             model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
#             model.fc = torch.nn.Linear(model.fc.in_features, 43)
#         else:
#             raise ValueError(f"Unsupported model name: {name}")
#         return model

#     def detect(self, image: Image.Image) -> TrafficDetectionResult:
#         """
#         Takes an input PIL image, runs inference, and returns structured traffic detection results.
#         """
#         # Preprocess the image and send it to the device
#         input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
#         # Forward
#         with torch.no_grad():
#             outputs = self.model(input_tensor)
#             probabilities = torch.nn.functional.softmax(outputs, dim=1)
#             confidence, predicted_class = torch.max(probabilities, dim=1)
        
#         # Return structured data that conforms to the conventions specified in types.py
#         traffic_signs = [
#             {
#                 "type": f"sign_class_{predicted_class.item()}",
#                 "distance_m": 10.0,
#                 "confidence": float(confidence.item())
#             }
#         ]
        
#         traffic_conditions = {
#             "traffic_light": "normal",
#             "road_condition": "clear"
#         }
        
#         return TrafficDetectionResult(
#             traffic_signs=traffic_signs,
#             traffic_conditions=traffic_conditions
#         )

from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from ..types import TrafficDetectionResult

# Mapping of GTSRB 43 class IDs to descriptive traffic sign labels
GTSRB_CLASSES = {
    0: "speed_limit_20", 1: "speed_limit_30", 2: "speed_limit_50", 3: "speed_limit_60",
    4: "speed_limit_70", 5: "speed_limit_80", 6: "end_speed_limit_80", 7: "speed_limit_100",
    8: "speed_limit_120", 9: "no_passing", 10: "no_passing_heavy", 11: "priority_next_intersection",
    12: "priority_road", 13: "yield", 14: "stop", 15: "no_vehicles", 16: "no_heavy_vehicles",
    17: "no_entry", 18: "general_danger", 19: "dangerous_curve_left", 20: "dangerous_curve_right",
    21: "double_curve", 22: "bumpy_road", 23: "slippery_road", 24: "road_narrows_right",
    25: "road_work", 26: "traffic_signals", 27: "pedestrians", 28: "children_crossing",
    29: "bicycles_crossing", 30: "ice_snow", 31: "wild_animals_crossing", 32: "end_speed_limit_zone",
    33: "turn_right_ahead", 34: "turn_left_ahead", 35: "ahead_only", 36: "go_straight_or_right",
    37: "go_straight_or_left", 38: "keep_right", 39: "keep_left", 40: "roundabout_mandatory",
    41: "end_of_no_passing", 42: "end_of_no_passing_heavy"
}


def estimate_distance(box_height_pixel: float, real_height_m: float = 0.6, focal_length_px: float = 800.0) -> float:
    """
    Estimate distance using the pinhole camera geometry model (D = f * H / h).
    :param box_height_pixel: Bounding box height in pixels
    :param real_height_m: Physical height of standard traffic signs (~0.6m)
    :param focal_length_px: Camera focal length in pixels
    """
    if box_height_pixel <= 0:
        return 0.0
    dist = (focal_length_px * real_height_m) / float(box_height_pixel)
    return round(float(dist), 2)


class TrafficDetector:
    """Interface for detecting traffic conditions and signs."""

    def __init__(self, model_name: str = "resnet18"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name.lower()

        # Load classification model architecture
        self.model = self._load_architecture(self.model_name)

        weights_path = (
            Path(__file__).parent.parent.parent.parent / "models" / "best_model.pth"
        )
        if weights_path.exists():
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device, weights_only=True))

        self.model.eval()
        self.model.to(self.device)

        # Image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _load_architecture(self, name: str):
        if "resnet" in name:
            from torchvision.models import ResNet18_Weights, resnet18
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
            model.fc = torch.nn.Linear(model.fc.in_features, 43)
            return model
        else:
            raise ValueError(f"Unsupported model name: {name}")

    def detect(self, image: Image.Image, conf_threshold: float = 0.5) -> TrafficDetectionResult:
        """
        Accepts a PIL image, performs sign detection, classification, distance estimation,
        and traffic condition evaluation.
        """
        # Convert PIL image to OpenCV BGR format for candidate proposal extraction
        cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        h_img, w_img, _ = cv_img.shape
        hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)

        # ----------------- 1. Sign Candidate Detection (ROI Extraction) -----------------
        # Segment red and blue regions (predominant traffic sign boundary colors)
        mask_r1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
        mask_r2 = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
        mask_b = cv2.inRange(hsv, np.array([100, 150, 50]), np.array([140, 255, 255]))
        mask = mask_r1 | mask_r2 | mask_b

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        traffic_signs = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # Filter noise and non-sign aspect ratios (signs are typically square or circular: 0.6 - 1.6)
            if w < 16 or h < 16 or not (0.6 < w / float(h) < 1.6):
                continue

            # Crop Region of Interest (ROI) and classify
            roi_pil = image.crop((x, y, x + w, y + h))
            input_tensor = self.transform(roi_pil).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(input_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted_class = torch.max(probs, dim=1)

            conf_val = float(confidence.item())
            if conf_val >= conf_threshold:
                pred_idx = predicted_class.item()
                dist = estimate_distance(box_height_pixel=h)
                sign_name = GTSRB_CLASSES.get(pred_idx, f"sign_class_{pred_idx}")

                # Strictly conform to Dict[str, Union[str, float]] schema
                traffic_signs.append({
                    "type": str(sign_name),
                    "distance_m": float(dist),
                    "confidence": float(round(conf_val, 4))
                })

        # ----------------- 2. Fallback for Pre-cropped Sign Inputs -----------------
        # If no contour is proposed but input is a small cropped image (e.g., benchmark unit tests)
        if not traffic_signs and (w_img < 150 and h_img < 150):
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted_class = torch.max(probs, dim=1)

            conf_val = float(confidence.item())
            if conf_val >= conf_threshold:
                pred_idx = predicted_class.item()
                dist = estimate_distance(box_height_pixel=h_img)
                sign_name = GTSRB_CLASSES.get(pred_idx, f"sign_class_{pred_idx}")

                traffic_signs.append({
                    "type": str(sign_name),
                    "distance_m": float(dist),
                    "confidence": float(round(conf_val, 4))
                })

        # ----------------- 3. Traffic Condition & Traffic Light Detection -----------------
        # Inspect the upper half of the FOV for active signal lights
        upper_hsv = hsv[0:int(h_img * 0.5), :]
        red_count = cv2.countNonZero(cv2.inRange(upper_hsv, np.array([0, 160, 160]), np.array([10, 255, 255])))
        green_count = cv2.countNonZero(cv2.inRange(upper_hsv, np.array([35, 100, 100]), np.array([85, 255, 255])))

        light_status = "red" if red_count > 80 else ("green" if green_count > 80 else "normal")
        road_status = "clear"

        # Strictly conform to Dict[str, str] schema
        traffic_conditions = {
            "traffic_light": light_status,
            "road_condition": road_status
        }

        return TrafficDetectionResult(
            traffic_signs=traffic_signs,
            traffic_conditions=traffic_conditions
        )