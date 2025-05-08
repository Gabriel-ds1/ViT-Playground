from dataclasses import dataclass, field
from typing import List

@dataclass
class TrainViTConfig:
    # Data & training
    data_dir: str = "./data"
    batch_size: int = 256
    epochs: int = 400
    img_size: int = 32
    val_split: float = 0.2

    # Optimizer & scheduler
    lr: float = 3e-4
    weight_decay: float = 0.05
    eta_min: float = 1e-6 # the lowest LR at plateau for scheduler
    early_stopping: int = 20

    # Model architecture
    use_amp: bool = True # mixed-precision
    patch_size: int = 4
    embed_dim: int = 768
    depth: int = 12
    mlp_ratio: float = 4.0
    num_heads: int = 8
    drop_rate: float = 0.2
    attn_drop_rate: float = 0.05

    # Logging & visualization
    project_name: str = "vit-cifar100"
    # ViT model visualizations -- options: "token_attn_maps", "cls_heatmap", "cls_dim_reduction", "hooks" -- or leave empty for none.
    # if you want to turn vit_visualizers off: vit_visualizers: List[str] = field(default_factory=lambda: [])
    vit_visualizers: List[str] = field(default_factory=lambda: ["token_attn_maps", "cls_heatmap", "cls_dim_reduction", "hooks"])
    visualize_every_n_epochs: int = 10
    visualize_layers: List[int] = field(default_factory=lambda:[0, 6, 11]) # which layers to visualize for "token_attn_maps", "cls_heatmap", and "cls_dim_reduction"
    save_npy: bool = False # Whether to save npy files alongside visualizations
    # whether to log tensorboard/weights and biases
    log_backend: str = "wandb"  # 'wandb', 'tensorboard', 'both', or leave empty for none

    # Checkpointing
    checkpoint_out_dir: str = "" # checkpoint output folder, leave empty for default checkpoints/run...
    save_every_n_epochs: int = 40 # epochs until new checkpoint save
    resume_from_checkpoint: bool = False
    resume_checkpoint_path: str = "" # path to ckpt file (.pth), leave empty for default