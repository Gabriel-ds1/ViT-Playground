import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset
from torchvision.transforms import AutoAugment, AutoAugmentPolicy

def get_cifar100_datasets(data_dir: str = './data', val_split: float = 0.1):
    """
    Prepare CIFAR-100 for training, validation, and testing.

    Args:
        data_dir: Directory to download/store CIFAR-100.
        val_split: Fraction of the train set to reserve for validation (0 disables).

    Returns:
        train_set, val_set, test_set
    """

    # CIFAR-100 dataset statistics
    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]

    # Data augmentations for training
    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.AutoAugment(policy=AutoAugmentPolicy.CIFAR10),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean, std),
                                          # cutout-like augmentation
                                          transforms.RandomErasing(p=0.2, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random')])
    
    # Normalization for validation/testing
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    # Download and load datasets
    train_full = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=test_transform)

    # Split train into train/val if val_split > 0
    if val_split and val_split > 0:
        val_size = int(len(train_full) * val_split)
        num_samples = len(train_full)
        indices = np.random.permutation(num_samples)
        val_size = int(num_samples * val_split)
        train_indices, val_indices = indices[val_size:], indices[:val_size]

        train_set = Subset(train_full, train_indices)
        val_dataset = datasets.CIFAR100(root=data_dir, train=True, download=False, transform=test_transform)
        val_set = Subset(val_dataset, val_indices)
    else:
        train_set = train_full
        val_set = None

    return train_set, val_set, test_set

def unnormalize(tensor):
    """
    Reverse CIFAR-100 normalization for visualization.

    Args:
        tensor (torch.Tensor): Normalized image tensor of shape (B, 3, H, W).

    Returns:
        torch.Tensor: Un-normalized image tensor in the [0,1] range.
    """
    mean = torch.tensor([0.5071, 0.4867, 0.4408], device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.2675, 0.2565, 0.2761], device=tensor.device).view(1, 3, 1, 1)
    img = tensor * std + mean
    return img.clamp(0.0, 1.0)