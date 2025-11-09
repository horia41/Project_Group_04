import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import os

def get_stats():
    """Calculates mean and std for normalization."""

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # train_data_path = os.path.join(script_dir, '..', 'current_data', '2019')
    # train_data_path = os.path.join(script_dir, '..', 'current_data', '2022')
    # train_data_path = os.path.join(script_dir, '..', 'DeepLearning_PlantDiseases-master', 'Scripts', 'PlantVillage_2019', 'train')
    train_data_path = os.path.join(script_dir, '..', 'DeepLearning_PlantDiseases-master', 'Scripts', 'PlantVillage_2022', 'train')

    if not os.path.exists(train_data_path):
        print(f"Error: Data path not found at {train_data_path}")
        return

    print(f"Calculating stats for dataset at: {train_data_path}")

    # --- 2. Setup Dataset & Loader ---
    stats_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    try:
        train_dataset = ImageFolder(root=train_data_path,
                                    transform=stats_transform)
        if not train_dataset:
            print("Error: No images found in the dataset folder.")
            return
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    loader = DataLoader(train_dataset,
                        batch_size=128,
                        shuffle=False,
                        num_workers=4,
                        pin_memory=True)

    # --- 3. Calculate Mean ---
    mean = torch.zeros(3)
    total_samples = 0

    with torch.no_grad():
        for images, _ in loader:
            batch_samples = images.size(0)
            images = images.view(batch_samples, images.size(1), -1)
            mean += images.mean(dim=2).sum(dim=0)
            total_samples += batch_samples

    mean /= total_samples
    print(f"Mean calculated: {mean}")

    # --- 4. Calculate Standard Deviation ---
    sum_squared_diff = torch.zeros(3)
    total_pixels = 0

    with torch.no_grad():
        for images, _ in loader:
            batch_samples, num_channels, height, width = images.shape
            mean_reshaped = mean.view(1, 3, 1, 1)
            squared_diff = (images - mean_reshaped).pow(2)
            sum_squared_diff += squared_diff.sum(dim=[0, 2, 3])
            total_pixels += (batch_samples * height * width)

    variance = sum_squared_diff / total_pixels
    std = torch.sqrt(variance)

    print(f"Standard Deviation calculated: {std}")
    print(f"mean = {mean.tolist()}")
    print(f"std  = {std.tolist()}")

    return mean.tolist(), std.tolist()

if __name__ == '__main__':
    get_stats()

# 2019 train entirely normalization params : mean 0.7553, 0.3109, 0.1059 || std 0.1774, 0.1262, 0.0863
# 2022 train entirely normalization params : mean 0.7083, 0.2776, 0.0762 || std 0.1704, 0.1296, 0.0815
# 2019 split normalization params : mean 0.7551, 0.3113, 0.1063 || std 0.1780, 0.1264, 0.0868
# 2022 split normalization params : mean 0.7074, 0.2772, 0.0759 || std 0.1713, 0.1298, 0.0812