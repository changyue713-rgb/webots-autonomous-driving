# Getting Started

## Installation

Clone the repository:

```bash
git clone https://github.com/USYD-ELEC5308/26-S2-Assignment1-Group1.git
cd 26-S2-Assignment1-Group1
```

Create and activate a virtual environment.

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the package and development dependencies:

```bash
pip install -e ".[dev]"
```

## Running Tests

Run the test suite with:

```bash
pytest tests/
```

## Running the Webots Demonstration

1. Open Webots.
2. Open `webots/worlds/city.wbt`.
3. Run the configured `my_controller`.
4. The autonomous vehicle will process camera input and use the perception and navigation modules during the simulation.

## Basic Python Usage

The autonomy modules can also be imported directly from the Python package.

For example:

```python
from elec5308_autonomy.perception.lane_detection import LaneDetector

detector = LaneDetector()
```

See the API Reference for the available modules, classes, and functions.

## Worked Usage Example

The following example shows how to use the lane detection module to process an input image.

```python
import numpy as np
from PIL import Image

from elec5308_autonomy.perception.lane_detection import LaneDetector

# Load an input image
image = np.array(Image.open("path/to/image.png").convert("RGB"))

# Create the lane detector
detector = LaneDetector()

# Detect lane boundaries and the driving area
result = detector.detect(image)

# Access the detection result
print("Lane detection result:")
print(result)
```

The LaneDetector accepts either a NumPy image array or a PIL image and returns a LaneDetectionResult. The result contains the detected lane and driving-area information produced by the perception pipeline.