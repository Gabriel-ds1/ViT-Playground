import os
import numpy as np
import re
import wandb
import shutil
import umap.umap_ as umap
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

def get_viz_save_path(base_dir, category, layer_idx=None, epoch=None, to_cls=None, filetype='png', extra=''):
    """
    Returns a Path for saving visualizations in the correct subfolder (creates dirs).
    Args:
        base_dir (Path): Base viz directory (e.g. checkpoints/runxxx/visualization)
        category (str): One of ['cls_attention', 'token_attention', 'cls_heatmap', 'cls_dim_reduction', 'hook_attention']
        layer_idx (int or str, optional): Layer index
        epoch (int, optional): Epoch number
        to_cls (bool, optional): For cls_attention, separate from/to cls
        filetype (str): Extension
        extra (str): Extra string for filename
    """
    if category == 'cls_attention':
        subcat = 'to_cls' if to_cls else 'from_cls'
        folder = base_dir / 'cls_attention' / subcat / f'layer{layer_idx}'
        if to_cls:
            fname = f'cls_to_tokens_attn_layer{layer_idx}_ep{epoch}{extra}.{filetype}'
        else:
            fname = f'cls_from_tokens_attn_layer{layer_idx}_ep{epoch}{extra}.{filetype}'
    elif category == 'token_attention':
        folder = base_dir / 'token_attention' / f'layer{layer_idx}'
        fname = f'token_attn_layer{layer_idx}_ep{epoch}{extra}.{filetype}'
    elif category == 'cls_heatmap':
        folder = base_dir / 'cls_heatmap' / f'layer{layer_idx}'
        fname = f'cls_heatmap_layer{layer_idx}_ep{epoch}{extra}.{filetype}'
    elif category == 'cls_dim_reduction':
        folder = base_dir / 'cls_dim_reduction' / f'layer{layer_idx}'
        fname = f'cls_dim_reduction_layer{layer_idx}_ep{epoch}{extra}.{filetype}'
    elif category == 'hook_attention':
        folder = base_dir / 'hook_attention'
        fname = f'hook_attention_ep{epoch}{extra}.{filetype}'
    else:
        raise ValueError(f"Unknown viz category {category}")

    folder.mkdir(parents=True, exist_ok=True)  # Make dirs if needed
    return folder / fname

class BackendVisualizer:
    """
    Unified backend logger/visualizer for training metrics and images supporting TensorBoard and/or Weights & Biases.

    This class provides a convenient interface to log scalar metrics, images, and model graphs 
    to supported backends during model training and evaluation. Supports TensorBoard, Weights & Biases (wandb), or both simultaneously.
    """
    def __init__(self, log_backend="both", log_dir="checkpoints/", project_name="vit-cifar100", config=None):
        self.log_backend = log_backend
        self.writer = None
        self.wandb_run = None

        if log_backend in ("tensorboard", "both"):
            self.writer = SummaryWriter(log_dir=log_dir)

        if log_backend in ("wandb", "both"):
            self.wandb_run = wandb.init(project=project_name, config=config)

    def log_scalars(self, metrics: dict, step: int):
        if self.writer and self.log_backend in ("tensorboard", "both"):
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, step)

        if self.wandb_run and self.log_backend in ("wandb", "both"):
            wandb.log(metrics, step=step)

    def log_images(self, tag: str, images, step: int):
        if self.writer and self.log_backend in ("tensorboard", "both"):
            self.writer.add_images(tag, images, step)

    def watch_model(self, model, criterion):
        if self.wandb_run and self.log_backend in ("wandb", "both"):
            wandb.watch(model, criterion, log="all")

    def close(self):
        if self.writer:
            self.writer.close()
        if self.wandb_run:
            wandb.finish()

class ViTVisualizer:
    """
    Utility class for creating and saving various Vision Transformer (ViT) visualizations,
    such as attention maps and CLS token projections, at specified training epochs.

    Args:
        model: The ViT model instance.
        depth (int): Number of transformer layers in the ViT model.
        filepath (str or Path): Base directory for saving outputs.
        epoch (int): Current epoch (for file naming).
        save_npy (bool, optional): Whether to export visualizations as .npy arrays.
    """
    # token_seqs list structure:
    # [0]: after patch embedding (CLS first index)
    # [1]: after pos_embed + dropout
    # [2]: after transformer block 0
    # ...
    # [depth]: after transformer block (depth-1)
    # [depth+1] or [-1]: after final LayerNorm (final model output before head)
    #
    # To get CLS token for a BLOCK, use:
    #   CLS = token_seqs[block_idx + 1][:, 0, :]
    # To get the FINAL CLS token (post-norm), use:
    #   CLS = token_seqs[-1][:, 0, :]

    def __init__(self, model, depth, filepath, epoch, save_npy=False):
        self.model = model
        self.depth = depth
        self.epoch = epoch
        self.filepath = filepath
        self.save_npy = save_npy

    def token_attention_maps(self, attn_maps, layers=None):
        """
        Visualizes attention maps from specified transformer blocks.

        Args:
            attn_maps (list of tensors): Attention maps from all blocks (from model forward).
            layers (list[int]): Block indices to visualize. Defaults to [1, depth//2, depth-1]
        """
        
        # if no layers provided, auto select first, middle and last layer
        if layers is None:
            layers = [1, self.depth // 2, self.depth - 1]  # e.g. early, middle, late

        for layer_idx in layers:
            # setup file output paths
            cls_from_save_path = get_viz_save_path(base_dir=self.filepath, category='cls_attention', layer_idx=layer_idx, epoch=self.epoch, to_cls=False)
            cls_to_save_path = get_viz_save_path(base_dir=self.filepath, category='cls_attention', layer_idx=layer_idx, epoch=self.epoch, to_cls=True)
            token_attention_save_path = get_viz_save_path(base_dir=self.filepath, category='token_attention', layer_idx=layer_idx, epoch=self.epoch)
            # attn_maps has shape [layer, batch, heads, tokens, tokens]
            block = attn_maps[layer_idx][:4].mean(dim=0) # grab all heads and the first image in batch
            # save
            save_cls_attention(f"layer{layer_idx}_att_FROM_cls", block, filepath=cls_from_save_path, epoch=self.epoch, to_cls=False)
            save_cls_attention(f"layer{layer_idx}_att_TO_cls", block, filepath=cls_to_save_path, epoch=self.epoch, to_cls=True)
            save_token_attention(f"layer{layer_idx}_attn_heads", block, filepath=token_attention_save_path, epoch=self.epoch)

    def cls_visualizers(self, token_seqs, labels, cls_heatmap=False, cls_dim_reduction=False, layers=None):
        """
        Save class token heatmaps and/or dimensionality reduction plots for selected layers.

        Args:
            token_seqs (list[Tensor]): Transformer token sequences at each layer.
            labels (Tensor): Class labels for the batch.
            cls_heatmap (bool): If True, generate CLS heatmaps.
            cls_dim_reduction (bool): If True, generate UMAP projections of CLS tokens.
            layers (list[int], optional): Layer indices to use.
        """
        # if no layers provided, auto select first, middle and last layer
        if layers is None:
            layers = [0, self.depth // 2, self.depth - 1]

        for layer_idx in layers:
            # setup file output paths
            cls_heatmaps_save_path = get_viz_save_path(base_dir=self.filepath, category='cls_heatmap', layer_idx=layer_idx, epoch=self.epoch)
            dim_reduction_save_path = get_viz_save_path(base_dir=self.filepath, category='cls_dim_reduction', layer_idx=layer_idx, epoch=self.epoch)
            # token_seqs[layer_idx]: (B, N, D), CLS is token 0
            #actual_layer_idx = layer_idx + 1  # offset due to patch_embed + pos_embed
            cls_embeds = token_seqs[-1][:, 0, :]
            #dim_red_cls_embeds = token_seqs[actual_layer_idx][:, 0, :]
            # save
            if cls_heatmap:
                save_cls_heatmaps(cls_embeds, labels, layer_idx, filepath=cls_heatmaps_save_path, epoch=self.epoch)
            if cls_dim_reduction:
                save_cls_dim_reduction(cls_embeds, labels=labels, layer_idx=layer_idx, filepath=dim_reduction_save_path, epoch=self.epoch)

    def hooks_attention(self):
        """
        Save feature activations from registered model hooks for debugging/analysis.
        """
        # setup file output paths
        hook_attn_save_path = get_viz_save_path(base_dir=self.filepath, category='hook_attention', epoch=self.epoch)
        # hooks
        hooks = {'patch_proj' : self.model.activations['patch_projection'],
            'after_pos_drop' : self.model.activations['after_pos_drop'],
            'post_norm' : self.model.activations['post_norm'],
            'attn_out_block1' : self.model.activations['attn_out_block1'],
            'attn_out_block6' : self.model.activations['attn_out_block6'],
            'attn_out_block12' : self.model.activations['attn_out_block11'],}
        # save
        save_hooks_attention(hooks, filepath=hook_attn_save_path)

def save_token_attention(name, attn_map, filepath, epoch):
    """
    Save all attention heads in a single image grid.

    attn_map: torch.Tensor of shape (H, N, N)
    """

    # Detach, move to cpu and convert to NumPy
    attn_np = attn_map.cpu().detach().numpy() if attn_map.device.type != "cpu" else attn_map.detach().numpy()
    assert attn_np.ndim == 3, "Expected shape (H, N, N) for multi-head attention"

    num_heads = attn_np.shape[0]

    # prepare number of columns and rows for output grid visualization
    cols = num_heads // 2
    rows = (num_heads + cols - 1) // cols

    fig, grid_axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = grid_axes.flatten()

    for i in range(num_heads):
        # grab grid axes index
        ax = axes[i]
        # plot image to grid axes
        im = ax.imshow(attn_np[i], cmap='viridis')
        ax.set_title(f"Head {i}")
        ax.axis("off")
        # add small color scale bar next to the plot
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # hide unused subplots if any
    for j in range(num_heads, len(axes)):
        axes[j].axis('off')
    
    fig.suptitle(f"{name} (epoch {epoch})", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(filepath, dpi=150)
    plt.close()

def save_cls_attention(name, attn_map, filepath, epoch, to_cls=False):
    """
    Save a CLS-centric attention map as a line plot.
      attn_map: torch.Tensor of shape (N, N) or (H, N, N)
      to_cls: if True, shows attention to CLS token (column 0); else from CLS token (row 0)
    """

    # Detach, move to cpu and convert to NumPy
    attn_np = attn_map.cpu().detach().numpy() if attn_map.device.type != "cpu" else attn_map.detach().numpy()

    # Average out attention maps for an average CLS plot
    attn_avg = attn_np.mean(axis=0)

    if to_cls:
        # Attention TO CLS
        cls_attn_avg = attn_avg[:, 0] # (N,) average over heads
        cls_attn_individual = attn_np[:, :, 0] # (H, N): every head's vector to CLS
    else:
        # Attention FROM CLS
        cls_attn_avg = attn_avg[0, :] # (N,) average over heads
        cls_attn_individual = attn_np[:, 0, :] # (H, N): every head's vector from CLS
    
    # Save average CLS attention plot
    plt.figure(figsize=(6,3))
    plt.plot(cls_attn_avg, marker='o')
    plt.title(f"{name} CLS {'receives' if to_cls else 'sends'} attention AVG (epoch {epoch})")
    plt.xlabel("Token index")
    plt.ylabel("Attention weight")
    plt.grid(True)
    plt.tight_layout()
    plt.figtext(0.5, -0.05, "Mean attention across 4 samples", ha='center', fontsize=9)
    plt.savefig(filepath, dpi=150)
    plt.close()

    # Save each individual CLS attention plot
    for i, vec in enumerate(cls_attn_individual):
        plt.figure(figsize=(6,3))
        plt.plot(vec, marker='o')
        plt.title(f"{name} CLS {'receives' if to_cls else 'sends'} attention (epoch {epoch})")
        plt.xlabel("Token index")
        plt.ylabel("Attention weight")
        plt.grid(True)
        plt.tight_layout()
        plt.figtext(0.5, -0.05, "Mean attention across 4 samples", ha='center', fontsize=9)
        plt.savefig(filepath, dpi=150)
        plt.close()
    
def save_cls_heatmaps(cls_embeds, labels, layer_idx, filepath, epoch):
    """
    Plot and save a heatmap of CLS token embeddings for all samples in a batch at a given layer.

    Args:
        cls_embeds (Tensor or np.ndarray): CLS embeddings, shape (batch, dim).
        labels (Tensor or np.ndarray): Class labels, shape (batch,).
        layer_idx (int): The index of the transformer layer.
        filepath (str or Path): Destination file path for saving the heatmap image.
        epoch (int): Current training epoch (for annotation purposes).
    """

    data = cls_embeds.cpu().detach().numpy() if cls_embeds.device.type != "cpu" else cls_embeds.detach().numpy()
    labels = labels.cpu().detach().numpy() if hasattr(labels, 'cpu') else np.array(labels)

    if len(np.unique(labels)) > 1:  # Only sort if more than one class, to avoid no-op
        order = np.argsort(labels)
        data = data[order]
        labels = labels[order]

    # set up plot
    fig, axes = plt.subplots(figsize=(12, 6))
    im = axes.imshow(data, aspect="auto", cmap="viridis")
    axes.set_title(f"CLS embeddings at layer {layer_idx} (epoch {epoch})\n(sorted by label)")
    axes.set_xlabel("Feature dimension")
    axes.set_ylabel("CLS embeddings for a single image")
    # add color scale bar next to the plot
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.04)
    cbar.set_label("Activation value", rotation=270, labelpad=15)
    # Draw horizontal lines at class boundaries
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            axes.axhline(i - 0.5, color='white', linewidth=0.5, alpha=0.6)
    
    batch, dim = data.shape
    axes.set_yticks(np.linspace(0, batch-1, min(batch, 10), dtype=int))
    axes.set_xticks(np.linspace(0, dim-1, min(dim, 12), dtype=int))
    axes.tick_params(axis='both', which='both', length=0)  # no tick marks

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()

def save_cls_dim_reduction(cls_embeds, labels, layer_idx, filepath, epoch):
    """
    Saves a uMap 2D projection of CLS embeddings with clustering metrics in the title.
    
    Args:
        cls_embeds (Tensor or np.ndarray): CLS embeddings, shape (batch, dim).
        labels (Tensor or np.ndarray): Class labels, shape (batch,).
        layer_idx (int): The index of the transformer layer.
        filepath (str or Path): Destination file path for saving the heatmap image.
        epoch (int): Current training epoch (for annotation purposes).
    """
    
    # cls_embeds should be (B, D)
    data = cls_embeds.cpu().detach().numpy() if cls_embeds.device.type != "cpu" else cls_embeds.detach().numpy()
    n_samples = data.shape[0]
    assert cls_embeds.shape[0] == labels.shape[0], "Batch dim mismatch between CLS tokens and labels!"
    
    # Labels should be (B,)
    label_data = labels.cpu().detach().numpy() if hasattr(labels, 'cpu') else np.array(labels)
    # sanity check
    assert label_data.ndim == 1 and label_data.shape[0] == data.shape[0], f"expeced labels length {data.shape[0]}, got {label_data.shape}"
    if n_samples < 4 or len(np.unique(label_data)) < 2:
        print("batch size must be greater than 2 and have more than one class to plot CLS dimensionality reduction visualization")
        return None

    pca = PCA(n_components=50)
    data_reduced = pca.fit_transform(data)

    # set up uMap projection
    z = umap.UMAP(n_components=2, random_state=42).fit_transform(data_reduced) # (N+1, 2)

    # check for degenerate output
    if np.std(z[:, 0]) < 1e-5 or np.std(z[:, 1]) < 1e-5:
        print("Degenerate uMap output (collapse to line/point), skipping plot.")
        return None
    
    fig, axes = plt.subplots(figsize=(6,6))
    axes.scatter(z[:,0], z[:,1], c=label_data, cmap='tab20', s=30)
    axes.set_title(f"uMap tokens @ layer {layer_idx} (epoch {epoch})\n")
    fig.text(0.5, 0.02, "Each color represents a class (e.g. dog, cat)", ha="center", va="bottom", fontsize=10)
    plt.tight_layout(rect=[0, 0.03, 0.75, 1])
    plt.savefig(filepath, bbox_inches="tight", dpi=150)
    plt.close()

def save_hooks_attention(hooks_dict, filepath, summarize_batch=False):
    """
    Plot and save visualizations for tensors recorded from model hooks.

    Args:
        hooks_dict (dict): Mapping from hook name to tensor (or tuple of tensors).
        filepath (str or Path): Output file path for saving the figure.
        summarize_batch (bool): If True, show mean over batch; otherwise plot first sample.
    """
    n = len(hooks_dict)
    if n == 0:
        return
    
    cols = n // 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*4))
    axes = axes.flatten()

    for ax, (name, tensor) in zip(axes, hooks_dict.items()):
        if isinstance(tensor, tuple):
            arr = tensor[0].detach().cpu().numpy() if tensor[0].device.type != "cpu" else tensor[0].detach().numpy()
        else:
            arr = tensor.detach().cpu().numpy() if tensor.device.type != "cpu" else tensor.detach().numpy()
        # plot first sample or mean across batch
        if arr.ndim > 1 and arr.shape[0] > 1:
            arr = arr.mean(axis=0) if summarize_batch else arr[0]
        if arr.ndim == 3:
            # Could be (H, N, N) attention or similar
            img = arr.mean(axis=0)
            im = ax.imshow(img, cmap='viridis')
            ax.set_title(f"{name} (mean over batch{'/heads' if arr.shape[0]>4 else ''})")
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
        elif arr.ndim == 2:
            im = ax.imshow(arr, aspect='auto', cmap='viridis')
            ax.set_title(f"{name} (N,D)")
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
        elif arr.ndim == 1:
            ax.plot(arr, marker='.')
            ax.set_title(f"{name} (D,)")
            ax.set_xlabel("Index")
            ax.grid(True)
        else:
            ax.axis('off')
            ax.set_title(f"{name} unsupported {arr.shape}")

    # hide unused
    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()

def collect_diverse_val_batch(val_loader, num_classes=100, samples_per_class=5, seed=42):
    """
    Collect a batch of images from the validation loader, containing up to N images per class.

    Args:
        val_loader (DataLoader): Validation data loader.
        num_classes (int): Number of different classes to collect.
        samples_per_class (int): Number of images per class to collect.

    Returns:
        diverse_images (list[Tensor]): List of image tensors.
        diverse_labels (list[int]): Corresponding class labels.
    """
    diverse_images = []
    diverse_labels = []
    label_to_images = {c: 0 for c in range(num_classes)}
    # Iterate once through val_loader to collect 1 image per class
    for images, labels in val_loader:
        for img, lbl in zip(images, labels):
            if label_to_images[lbl.item()] < samples_per_class:
                diverse_images.append(img.unsqueeze(0))
                diverse_labels.append(lbl)
                label_to_images[lbl.item()] += 1
            # Check if finished collecting for all classes
            if all([label_to_images[c] == samples_per_class for c in label_to_images]):
                break
        if all([label_to_images[c] == samples_per_class for c in label_to_images]):
            break
    return diverse_images, diverse_labels


def get_latest_viz_zip(checkpoint_dir):
    """
    Find and return the most recent (highest epoch) visualization zip file in the checkpoint directory.

    Args:
        checkpoint_dir (str or Path): Directory containing visualization zips.

    Returns:
        str or None: Path to the latest viz zip, or None if none found.
    """
    best_path = os.path.join(checkpoint_dir, "visualization_best.zip")
    if os.path.exists(best_path):
        return best_path  # Return immediately if best.zip exists
    
    # if best.zip not found, look for epoch-numbered zips
    zips = [f for f in os.listdir(checkpoint_dir) if re.match(r"visualization_ep\d+\.zip", f)]
    if not zips:
        return None  # No viz zips found at all
    
    # Sort by epoch number, highest first
    zips.sort(key=lambda x: int(re.search(r"ep(\d+).zip", x).group(1)), reverse=True)
    return os.path.join(checkpoint_dir, zips[0])


def save_visualization_zip(visualizations_dir, epoch):
    """
    Archive all files in the visualizations directory into a zip file for the current epoch.

    Args:
        visualizations_dir (str or Path): Directory to archive.
        epoch (int): Current epoch for naming the zip file.

    Returns:
        str: Path to the created zip file.
    """
    zip_path = str(visualizations_dir.parent / f"visualization_ep{epoch}.zip")
    shutil.make_archive(base_name=zip_path.replace('.zip', ''), format='zip', root_dir=str(visualizations_dir))
    return zip_path
