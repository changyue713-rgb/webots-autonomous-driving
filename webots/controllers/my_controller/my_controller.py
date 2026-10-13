"""Webots controller integrating lane detection and basic path following."""

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from vehicle import Driver

# Project root:
# project/
# ├── src/
# └── webots/controllers/my_controller/my_controller.py
project_root = Path(__file__).resolve().parents[3]
project_src = project_root / "src"

if str(project_src) not in sys.path:
    sys.path.insert(0, str(project_src))

from elec5308_autonomy.navigation.path_following import PathFollower
from elec5308_autonomy.navigation.speed_control import SpeedController
from elec5308_autonomy.navigation.steering import SteeringController
from elec5308_autonomy.perception.lane_detection import LaneDetector
from elec5308_autonomy.perception.object_detection import ObjectDetector
from elec5308_autonomy.perception.object_tracker import ObjectTracker
from elec5308_autonomy.perception.segmentation import SemanticSegmenter
from elec5308_autonomy.perception.traffic_detection import TrafficDetector

# Webots driving parameters.
# Keep the first integration test slow so that we can check the steering safely.
MAX_SPEED = 45.0

# Object detection is far slower than lane detection, so running it on every
# frame would stall the control loop. Run it every Nth frame instead.
DETECTION_INTERVAL = 30


def camera_to_pil(camera):
    """Convert a Webots camera image to a PIL RGB image."""
    image_bytes = camera.getImage()

    if image_bytes is None:
        return None

    width = camera.getWidth()
    height = camera.getHeight()

    image = np.frombuffer(
        image_bytes, dtype=np.uint8
    ).reshape((height, width, 4))

    # Webots returns BGRA. Convert to RGB and discard alpha.
    rgb = image[:, :, [2, 1, 0]]
    return Image.fromarray(rgb)


driver = Driver()
timestep = int(driver.getBasicTimeStep())

camera = driver.getDevice("camera")
camera.enable(timestep)

lane_detector = LaneDetector()
object_detector = ObjectDetector()
object_tracker = ObjectTracker()
segmentation_model = SemanticSegmenter()
path_follower = PathFollower()
steering_controller = SteeringController()
speed_controller = SpeedController()
traffic_detector = TrafficDetector()

print("my_controller: lane-following controller started")
print(f"Camera resolution: {camera.getWidth()}x{camera.getHeight()}")

frame_count = 0

while driver.step() != -1:
    frame_count += 1

    image = camera_to_pil(camera)

    if image is None:
        continue

    # Object detection and tracking. Run periodically rather than every frame
    # so the steering loop stays responsive. Kept outside the lane-following
    # try/except: a perception failure here should not stop the car.
    if frame_count % DETECTION_INTERVAL == 0:
        detection_result = object_detector.detect(image)
        track_ids = object_tracker.update(detection_result)

        if detection_result.objects:
            tracked = ", ".join(
                f"{label}#{track_id} ({score:.2f})"
                for label, track_id, score in zip(
                    detection_result.objects,
                    track_ids,
                    detection_result.confidence,
                )
            )
            print(f"frame={frame_count}, tracked: {tracked}")

        # Traffic sign detection
        traffic_result = traffic_detector.detect(image)
        if traffic_result and traffic_result.traffic_signs:
            print(f"frame={frame_count}, traffic_signs: {traffic_result.traffic_signs}")

         # >>> ADDED: call semantic segmentation and print its output 
        segmentation_result = segmentation_model.segment(np.array(image))
        print(f"frame={frame_count}, segmentation classes: {np.unique(segmentation_result.segmentation_map)}")


    try:
        # 1. Detect the left/right lane boundaries and driving area.
        lane_result = lane_detector.detect(image)
        if frame_count % 50 == 0:
            print(
                f"Lane result: "
                f"left={len(lane_result.left_lane)}, "
                f"right={len(lane_result.right_lane)}"
            )

        # 2. Generate a centre path between the two lane boundaries.
        navigation_info = path_follower.follow(lane_result)

        # 3. Convert the path direction into a steering angle.
        steering_result = steering_controller.calculate_steering(
            navigation_info
        )

        # 4. Select a speed according to the amount of turning.
        speed_result = speed_controller.calculate_speed(
            navigation_info
        )

        # SpeedController returns a normalised level:
        # 1.0 = straight, 0.7 = moderate turn, 0.4 = sharp turn.
        raw_steering = float(steering_result.steering_angle)
        steering_angle = float(np.clip(raw_steering, -0.5, 0.5))

        cruising_speed = MAX_SPEED * float(speed_result.speed)

        driver.setSteeringAngle(steering_angle)
        driver.setCruisingSpeed(cruising_speed)

        # Print only occasionally to avoid flooding the Webots console.
        if frame_count % 50 == 0:
            print(
                f"frame={frame_count}, "
                f"steering={steering_result.steering_angle:.3f}, "
                f"speed={cruising_speed:.1f}"
            )

    except ValueError as exc:
        # If the lane/path is temporarily invalid, stop rather than
        # sending an unsafe steering command.
        driver.setCruisingSpeed(0.0)
        driver.setSteeringAngle(0.0)

        if frame_count % 20 == 0:
            print(f"Lane/path unavailable: {exc}")