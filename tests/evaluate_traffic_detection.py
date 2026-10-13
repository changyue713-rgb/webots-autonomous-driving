# import time
# import torch
# import torchvision
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader
# import os

# def evaluate_models():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Equipment used for the assessment.: {device}")

#     # Prepare the test set
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#     data_dir = os.path.join(os.getcwd(), "data")
    
#     if not os.path.exists(data_dir):
#         print("No 'data' directory was detected. Please ensure that the GTSRB dataset has been correctly downloaded to the 'data' folder in the root directory.")
#         return

#     test_dataset = torchvision.datasets.GTSRB(root=data_dir, split="test", download=False, transform=transform)
#     test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

#     # Define the list of models to be compared
#     import torchvision.models as models

#     models_to_test = {
#         "ResNet-18": models.resnet18(pretrained=True),
#     }

#     results = {}

#     for name, model in models_to_test.items():
#         print(f"\nEvaluating.: {name} ...")
#         model.fc = torch.nn.Linear(model.fc.in_features, 43)
#         model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
#     model.to(device)
#     model.eval()

#     correct = 0
#     total = 0
#     start_time = time.time()

#     with torch.no_grad():
#         for images, labels in test_loader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             _, predicted = outputs.max(1)
#             total += labels.size(0)
#             correct += predicted.eq(labels).sum().item()

#     elapsed_time = time.time() - start_time
#     accuracy = 100. * correct / total
#     avg_latency = (elapsed_time / total) * 1000 

#     results[name] = {
#             "Accuracy": accuracy,
#             "Latency_ms": avg_latency
#         }
#     print(f"-> {name} result: accuracy = {accuracy:.2f}%, Average reasoning time = {avg_latency:.2f} ms/image")

#     print("\n========== Final model evaluation comparison==========")
#     for name, metrics in results.items():
#         print(f"[{name}] accuracy: {metrics['Accuracy']:.2f}% | delay: {metrics['Latency_ms']:.2f} ms")

# if __name__ == "__main__":
#     evaluate_models()

import sys
from pathlib import Path

from PIL import Image

# Add src directory to path so elec5308_autonomy can be imported directly
sys.path.append(str(Path(__file__).parent / "src"))

from elec5308_autonomy.perception.traffic_detection import TrafficDetector
from elec5308_autonomy.types import TrafficDetectionResult


def evaluate_sample_images(detector: TrafficDetector, sample_dir: Path):
    """
    Evaluates detector performance across test images and prints formatted results.
    """
    supported_exts = [".ppm", ".png", ".jpg", ".jpeg"]
    image_files = [p for p in sample_dir.glob("**/*") if p.suffix.lower() in supported_exts]

    if not image_files:
        print(f"No test images found in '{sample_dir}'.")
        return

    print(f"\nRunning evaluation on {len(image_files)} sample image(s) from {sample_dir}...\n")
    print(f"{'Image Name':<25} | {'Detected Signs (Type / Dist / Conf)':<45} | {'Conditions'}")
    print("-" * 90)

    for img_path in image_files[:20]:  # Evaluate up to 20 samples to keep output readable
        try:
            image = Image.open(img_path).convert("RGB")
            result: TrafficDetectionResult = detector.detect(image, conf_threshold=0.5)

            # Format signs output string
            if result.traffic_signs:
                signs_desc = "; ".join([
                    f"{s['type']} ({s['distance_m']}m, {s['confidence']:.2f})"
                    for s in result.traffic_signs
                ])
            else:
                signs_desc = "None"

            # Format conditions output string
            cond_desc = f"Light: {result.traffic_conditions.get('traffic_light')}, Road: {result.traffic_conditions.get('road_condition')}"

            print(f"{img_path.name:<25} | {signs_desc:<45} | {cond_desc}")

        except Exception as e:
            print(f"{img_path.name:<25} | Error during inference: {e}")

    print("-" * 90)


def main():
    print("Initializing TrafficDetector...")
    detector = TrafficDetector(model_name="resnet18")

    # Priority paths for test samples
    test_paths = [
        Path("data/sample_images"),
        Path("data/gtsrb/GTSRB/Final_Test"),
        Path("data/gtsrb/GTSRB/Training/00001"),  # Fallback to verify pipeline on a class folder
    ]

    target_dir = None
    for p in test_paths:
        if p.exists():
            target_dir = p
            break

    if target_dir:
        evaluate_sample_images(detector, target_dir)
    else:
        print("No evaluation dataset path found. Please provide test images under 'data/sample_images'.")


if __name__ == "__main__":
    main()