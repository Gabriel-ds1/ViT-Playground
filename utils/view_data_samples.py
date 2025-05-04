from ..datasets.cifar100 import unnormalize
from ..datasets.loaders import get_cifar100_dataloaders
import matplotlib.pyplot as plt
import torchvision
from torchvision.datasets import CIFAR100

# Helper function to show a batch of images
def show_batch(loader, class_names=None, title='Batch'):
    # Get one batch
    images, labels = next(iter(loader))
    
    # Unnormalize
    images = unnormalize(images)
    
    # Make a grid
    img_grid = torchvision.utils.make_grid(images, nrow=4)
    
    # Convert from tensor to numpy
    npimg = img_grid.permute(1, 2, 0).cpu().numpy()

    # Plot
    plt.figure(figsize=(16, 16))
    plt.imshow(npimg)
    if class_names is not None:
        # Show the class names for the batch
        label_names = [class_names[label.item()] for label in labels]
        plt.title(f"{title}\n" + ", ".join(label_names))
    else:
        plt.title(title)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_cifar100_dataloaders(data_dir='./data', batch_size=16, val_split=0.2)
    class_names = CIFAR100('./data', train=True, download=False).classes
    show_batch(train_loader, class_names, title='Train Loader Batch')
    show_batch(val_loader, class_names, title='Validation Loader Batch')
    show_batch(test_loader, class_names, title='Test Loader Batch')