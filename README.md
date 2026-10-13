# ELEC5308 Assignment 1: Autonomous Perception and Navigation System

An autonomous vehicle perception and navigation system developed for **ELEC5308**, combining computer vision pipelines and deep learning models to enable lane following, obstacle detection/tracking, traffic sign recognition, and semantic scene understanding in the **Webots** simulator.

---

## 👥 Team Members

* **[Muhammad Usman Khan]** - [540756177]
* **[Zhiqi Liu]** - [550223481]
* **[Saifullah Tawsif]** - [530820172]
* **[Yue Chang]** - [560292484]
* **[Alaa Aldirani]** - [490627356]

---

## 📖 Project Description

This repository provides `elec5308_autonomy`, a modular, simulator-independent library paired with Webots simulation controllers. The core system architecture includes:

* **Perception Modules (`elec5308_autonomy.perception`)**:
  * **Lane Detection (`lane_detection.py`)**: HSV color thresholding (yellow and white markings), morphological filtering, trapezoidal ROI masking, Canny edge detection, and Hough transforms to identify lane lines and drivable road corridors.
  * **Object Detection (`object_detection.py`)**: Faster R-CNN with a ResNet-50-FPN backbone for real-time traffic obstacle and vehicle detection.
  * **Object Tracking (`object_tracker.py`)**: Centroid-based Hungarian matching algorithm (`scipy.optimize.linear_sum_assignment`) with missed-detection aging for consistent track IDs across frames.
  * **Traffic Sign Recognition (`traffic_detection.py`)**: Color-based sign proposal and distance estimation using a pinhole camera model, classified using a fine-tuned ResNet-18 model across 43 GTSRB sign classes and traffic lights.
  * **Semantic Segmentation (`segmentation.py` & `unet_model.py`)**: U-Net architecture with a ResNet-18 backbone for pixel-wise drivable area and obstacle categorization.
* **Navigation & Control (`elec5308_autonomy.navigation`)**:
  * **Path Following (`path_following.py`)**: Generates navigation waypoints along the center path between detected lane boundaries.
  * **Steering Control (`steering.py`)**: Proportional steering calculation with exponential smoothing to suppress frame-to-frame angular oscillations.
  * **Speed Control (`speed_control.py`)**: Adaptive curvature-based speed scaling.
* **Simulation Adapter (`elec5308_autonomy.adapters`)**:
  * Decouples the core autonomy logic from the simulator backend (`SimulatorAdapter`), implemented via `WebotsAdapter` and controllers in `webots/controllers/`.

---

## 🛠️ Installation Steps

### Prerequisites
* Python `>= 3.10`
* [Webots R2025a](https://cyberbotics.com/)
* CUDA-enabled GPU (optional, but recommended for deep learning inference)

### 1. Clone the Repository
```bash
git clone https://github.com/USYD-ELEC5308/26-S2-Assignment1-Group1.git
cd 26-S2-Assignment1-Group1
```

### 2. Create and Activate a Virtual Environment
```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate

# Windows (Command Prompt / PowerShell)
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Package and Dependencies
Install the package in editable mode along with development dependencies:
```bash
pip install --upgrade pip
pip install -e ".[dev]"
```

### 4. Download Trained Weights & Datasets (Optional)
If training or running evaluation benchmarks locally:
```bash
# Download the GTSRB dataset for traffic sign classification
python download_dataset.py
```
Ensure required model checkpoints (`models/best_model.pth` and `models/segmentation_best_model.pth`) are placed in the `models/` directory.

---

## 🚀 Quick-Start Example

### 1. Run Unit and Pipeline Tests
Verify that all core perception and navigation modules pass tests:
```bash
pytest tests/
```

### 2. Evaluate Traffic Sign Detection Pipeline
Run inference over test samples using the pre-trained sign detector:
```bash
python evaluate_traffic_detection.py
```

### 3. Run the Full Closed-Loop Simulation in Webots
1. Launch **Webots**.
2. Open the world file: `webots/worlds/city.wbt`.
3. The BMW X5 vehicle is configured to run `my_controller`.
4. Click **Play / Run** in Webots. The autonomous vehicle will:
   * Process camera frames via `LaneDetector` to steer and follow lanes.
   * Periodically trigger `ObjectDetector`, `ObjectTracker`, and `TrafficDetector` to track obstacles and detect traffic signs.
   * Adjust driving speed and steering dynamically.

---

## 📂 Repository Structure

```text
.
├── ai-journal/                 # AI interaction and planning journals
├── data/                       # Local dataset directory (GTSRB, etc.)
├── models/                     # Trained model checkpoints (.pth)
├── src/
│   └── elec5308_autonomy/      # Core autonomy library
│       ├── adapters/           # Simulator interface & Webots adapter
│       ├── navigation/         # Steering, speed, and path following
│       ├── perception/         # Lane, object, tracker, traffic & segmentation
│       ├── __init__.py         # Package initialization
│       └── types.py            # Shared data contracts and dataclasses
├── tests/                      # Pytest test suite
├── webots/
│   ├── controllers/
│   │   ├── my_controller/      # Main closed-loop driving controller
│   │   └── object_detector_test/ # Standalone frame capture test harness
│   └── worlds/
│       └── city.wbt            # Webots urban simulation environment
├── download_dataset.py         # GTSRB dataset downloader script
├── pyproject.toml              # Build configurations & package dependencies
├── requirements.txt            # Dependency specification file
└── train_traffic_detection.py  # Training script for traffic sign classifier

---

## 📚 Documentation

The project provides automatically generated HTML documentation for the public Python API.

The documentation includes:

* Project overview
* Getting Started guide
* API reference
* Public classes and functions
* Usage examples

The documentation can be built locally using:

```bash
mkdocs build --strict
```

To preview the documentation locally:

```bash
mkdocs serve
```

The generated HTML documentation will be placed in the `site/` directory.
