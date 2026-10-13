"""Throwaway test harness: grab several consecutive camera frames from the BmwX5
in city.wbt, run each through our ObjectDetector, then feed the results into our
ObjectTracker to check whether track IDs stay consistent across frames.

This is NOT the real simulator adapter. Its only job is to prove that detection
and tracking work together on genuine Webots frames, not just synthetic test data.
"""

import sys
from pathlib import Path


def find_repo_root(start: Path) -> Path:
    """Walk upwards from `start` until we find the folder containing pyproject.toml.

    Webots launches this file directly (not via `pip install`), so Python does not
    know where our `elec5308_autonomy` package lives. We locate the repo root by
    looking for pyproject.toml, then add its `src/` folder to the import path.
    """
    for parent in [start, *start.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not find repo root (no pyproject.toml above this file)")


REPO_ROOT = find_repo_root(Path(__file__).resolve())
sys.path.insert(0, str(REPO_ROOT / "src"))

# `controller` is Webots' own Python module (found via the Webots Python setup).
# Our own code and dependencies.
from controller import Robot
from PIL import Image, ImageDraw

from elec5308_autonomy.perception.object_detection import ObjectDetector
from elec5308_autonomy.perception.object_tracker import ObjectTracker

# How many simulation steps to let pass before we trust the rendered image.
# Set high enough that the car has fully settled on its suspension, otherwise
# the camera is still bobbing and detections drift between frames.
WARMUP_STEPS = 50

# How many separate frames to capture and run through detection + tracking.
NUM_FRAMES_TO_CAPTURE = 4

# Simulation steps to advance between each captured frame, giving objects a
# small chance to move so tracking has something meaningful to match.
STEPS_BETWEEN_FRAMES = 5


def main() -> None:
    # 1. Take control of the simulation.
    robot = Robot()

    # 2. The world's basic time step, in milliseconds.
    timestep = int(robot.getBasicTimeStep())

    # 3. Grab the camera (default name "camera", verified against city.wbt).
    camera = robot.getDevice("camera")
    camera.enable(timestep)

    # 4. Let the renderer settle before trusting any frame.
    for _ in range(WARMUP_STEPS):
        if robot.step(timestep) == -1:
            return

    # 5. Create the detector and tracker once, reused across all frames.
    #    Reusing one ObjectTracker instance across frames is what lets it
    #    remember previous positions and assign consistent IDs.
    detector = ObjectDetector()
    tracker = ObjectTracker()

    width = camera.getWidth()
    height = camera.getHeight()

    # 6. Capture, detect, and track across multiple frames.
    for frame_num in range(NUM_FRAMES_TO_CAPTURE):
        raw = camera.getImage()
        image = Image.frombytes("RGBA", (width, height), raw, "raw", "BGRA").convert("RGB")

        result = detector.detect(image)
        track_ids = tracker.update(result)

        print(f"--- Frame {frame_num} ---")
        print(f"objects:    {result.objects}")
        print(f"track_ids:  {track_ids}")
        print(f"boxes:      {result.bounding_boxes}")
        print(f"confidence: {result.confidence}")

        # 7. Save the frame with detection boxes drawn on it, so we can look at
        #    what the detector actually latched onto rather than guessing from
        #    coordinates alone.
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        for label, box, score in zip(result.objects, result.bounding_boxes, result.confidence):
            draw.rectangle(box, outline="red", width=2)
            draw.text((box[0], box[1] - 10), f"{label} {score:.2f}", fill="red")
        annotated.save(str(REPO_ROOT / f"frame_{frame_num}.png"))

        for _ in range(STEPS_BETWEEN_FRAMES):
            if robot.step(timestep) == -1:
                return

    print("Done capturing frames.")


if __name__ == "__main__":
    main()