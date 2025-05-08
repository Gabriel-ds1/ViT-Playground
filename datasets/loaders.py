from torch.utils.data import DataLoader
from .cifar100 import get_cifar100_datasets

def get_cifar100_dataloaders(data_dir: str = './data', batch_size: int = 128, num_workers: int = 4, val_split: float = 0.1):
    """
    Prepare CIFAR-100 DataLoaders for training, validation, and testing.

    Args:
        data_dir: Directory to download/store CIFAR-100.
        batch_size: Batch size for all Dataloaders.
        num_workers: number of subprocesses for data loading.
        val_split: Fraction of the train set to reserve for validation (0 disables).

    Returns:
        train_loader, val_loader, test_loader
    """
    # Build raw dataset splits
    train_set, val_set, test_set = get_cifar100_datasets(data_dir = data_dir, val_split = val_split)

    # Create DataLoader objects
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = None
    if val_set is not None:
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
