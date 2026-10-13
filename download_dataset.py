import os

import torchvision
import torchvision.transforms as transforms

data_dir = os.path.join(os.getcwd(), "data")

print("downloading GTSRB dataset...")
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])


train_dataset = torchvision.datasets.GTSRB(
    root=data_dir,
    split="train",
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.GTSRB(
    root=data_dir,
    split="test",
    download=True,
    transform=transform
)

print(f"Download complete. train sample: {len(train_dataset)}, test sample: {len(test_dataset)}")