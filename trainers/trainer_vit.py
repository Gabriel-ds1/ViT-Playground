import os
import random
import time
import numpy as np
import torch
import wandb
from pathlib import Path
import zipfile
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast
from dataclasses import fields
from datasets.loaders import get_cifar100_dataloaders
from datasets.cifar100 import unnormalize
from network.transformers import ViT
from network import utils
from utils.common_utils import create_checkpoint_path
from loggers.log import setup_logger
from loggers.visualizers import BackendVisualizer, ViTVisualizer, collect_diverse_val_batch, save_visualization_zip, get_latest_viz_zip
from configs.config import TrainViTConfig


class TrainerViT:
    """
    Trainer for Vision Transformer (ViT) models with support for reproducibility, mixed precision,
    logging, visualizations, checkpointing, and training with early stopping.

    Args:
        config (TrainViTConfig): Training and model hyperparameters.

    Attributes:
        device (torch.device): Device used for training.
        out_dir (Path): Output directory for checkpoints and logs.
        logger: Logging object for file and console output.
        backend_vis: Backend visualizer (e.g., wandb/tensorboard).
    """
    def __init__(self, config: TrainViTConfig):
        """
        Initialize reproducibility, device, data, logging directories, and visualization setups.
        """
        # setup configs -- see configs/config.py for user arguments
        self.cfg = config
        # unpack and set all config args
        for f in fields(config):
            setattr(self, f.name, getattr(config, f.name))

        # Set seeds for reproducibility
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Select device: GPU if available, otherwise CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Set mixed precision if use_amp config == True
        if self.use_amp:
            self.scalar = GradScaler()

        # Create output checkpoint folder (or default)
        self.out_dir = Path(self.checkpoint_out_dir or create_checkpoint_path())
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # sanity checks
        if not 0 < self.val_split < 1: # val_split
            raise ValueError("val_split must be between 0 and 1")
        if (not isinstance(self.patch_size, int) or self.patch_size <= 0 or self.patch_size > self.img_size or (self.img_size % self.patch_size) != 0):
            raise ValueError(f"patch_size must be a positive integer <= ({self.img_size})" f" and must evenly divide it, but got {self.patch_size}")
        if not isinstance(self.save_every_n_epochs, int) or self.save_every_n_epochs < 0:
            raise ValueError(f"save_every_n_epochs must be a non-negative integer, got {self.save_every_n_epochs}")
        if not isinstance(self.visualize_every_n_epochs, int) or self.visualize_every_n_epochs < 1:
            raise ValueError(f"visualize_every_n_epochs must be >=1, got {self.visualize_every_n_epochs}")
        SUPPORTED_VISUALIZERS = {"token_attn_maps", "cls_heatmap", "cls_dim_reduction", "hooks"}
        for v in self.vit_visualizers:
            if v not in SUPPORTED_VISUALIZERS:
                raise ValueError(f"Unsupported visualizer {v}")

        # If resuming but no resume dir specified, point it at out_dir
        if self.resume_from_checkpoint and not self.resume_checkpoint_path:
            self.resume_checkpoint_path = str(self.out_dir)

        # Visualizations subfolder
        self.visualizations_dir = self.out_dir / "visualization"
        self.visualizations_dir.mkdir(exist_ok=True)

        # set up backend logging (Tensorboard/Weights and Biases)
        if self.log_backend is not None:
            self.backend_vis = BackendVisualizer(log_backend = self.log_backend, log_dir=self.visualizations_dir, project_name=self.project_name, 
                                         config = {"learning_rate": self.lr, "batch_size": self.batch_size})

        # Logger for console/file output
        self.logger = setup_logger(self.out_dir)

    def setup_dataloaders(self):
        """
        Loads CIFAR-100 dataset and creates PyTorch DataLoaders for train, val, and test splits.
        Also collects a diverse validation sample batch for visualizations.
        """
        # Data loaders
        self.train_loader, self.val_loader, self.test_loader = get_cifar100_dataloaders(data_dir=self.data_dir, batch_size=self.batch_size, 
                                                                                        val_split=self.val_split)
        # Collect diverse labels and samples for t-sne visualization
        diverse_images, diverse_labels = collect_diverse_val_batch(self.val_loader, num_classes=100, samples_per_class=2) # Stack to make a batch, using 100 classes for CIFAR-100
        self.diverse_val_batch = (torch.cat(diverse_images, 0), torch.tensor(diverse_labels))
        
    def setup_backend_visualization(self):
        """
        Sets up backend logging (wandb or tensorboard) for monitoring gradients or image samples.
        """
        if "wandb" in self.log_backend:
            wandb.watch(self.model, self.criterion, log="all")
        if "tensorboard" in self.log_backend:
            self.sample_batch = next(iter(self.train_loader))
    
    def output_model_visualizations(self, epoch):
        """
        Runs and saves ViT visualizations (attention maps, heatmaps, etc.) for the current epoch.
        Collects attention maps, tokens, and hooks if configured.
        
        Args:
            epoch (int): The current epoch number.
        """
        # Output ViT visualizations
        if (epoch == 1 or epoch % self.visualize_every_n_epochs == 0) and self.vit_visualizers:
            with torch.no_grad():
                print("Exporting ViT visualizers...")
                diverse_images, diverse_labels = (t.to(self.device) for t in self.diverse_val_batch)
                vit_visualization = ViTVisualizer(model=self.model, depth=self.depth, filepath = self.visualizations_dir, epoch=epoch, save_npy=self.save_npy)
                # Run forward pass to grab attention maps and tokens
                diverse_outputs, diverse_attn_maps, diverse_tokens = self.model(diverse_images, return_attn=True, return_tokens=True)
                attn_maps = [a.cpu() for a in diverse_attn_maps]
                tokens = [t.cpu() for t in diverse_tokens]
                labels_cpu = diverse_labels.cpu()
                if "token_attn_maps" in self.vit_visualizers:
                    vit_visualization.token_attention_maps(attn_maps, layers=self.visualize_layers)
                if "cls_heatmap" in self.vit_visualizers or "cls_dim_reduction" in self.vit_visualizers:
                    vit_visualization.cls_visualizers(tokens, labels=labels_cpu, cls_heatmap="cls_heatmap" in self.vit_visualizers, 
                                                      cls_dim_reduction="cls_dim_reduction" in self.vit_visualizers, layers=self.visualize_layers)
                if "hooks" in self.vit_visualizers:
                    vit_visualization.hooks_attention()
                del diverse_images, labels_cpu, diverse_tokens, diverse_attn_maps, diverse_outputs
                torch.cuda.empty_cache()

    def setup_model(self):
        """
        Instantiates the ViT model, criterion, optimizer, and learning rate scheduler.
        Logs basic info about the device and model.
        """
        self.logger.info("device: %s", str(self.device))
        # Initialize model
        self.model = ViT(img_size=self.img_size, patch_size=self.patch_size, in_channels=3, num_classes=100, embed_dim=self.embed_dim, depth=self.depth, 
                         num_heads=self.num_heads, mlp_ratio=self.mlp_ratio, drop_rate=self.drop_rate, attn_drop_rate=self.attn_drop_rate, use_conv_stem=True, conv_stem_dim=64).to(self.device)
        
        # Loss, optimizer, scheduler
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # Reduce LR when validation loss plateaus
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=self.cosine_eta_min)
    
    def resume_training(self):
        """
        Loads checkpointed state from disk and restores model, optimizer, scheduler, and
        visualization artifacts for resuming training.
        """
        if not os.path.exists(self.resume_checkpoint_path):
            raise FileNotFoundError(f"No checkpoint found at {self.resume_checkpoint_path}")
        self.setup_dataloaders()
        self.setup_model()

        checkpoint = torch.load(self.resume_checkpoint_path, map_location=self.device)

        # load in visualizations checkpoint
        if self.vit_visualizers:
            resume_ckpt_dir, _ = os.path.split(self.resume_checkpoint_path)
            viz_zip_path = get_latest_viz_zip(resume_ckpt_dir)
            if viz_zip_path:
                with zipfile.ZipFile(viz_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.visualizations_dir)
                self.logger.info(f"Resumed visualizations from {viz_zip_path}")
            else:
                self.logger.info("No previous visualization zip found. Will start fresh.")

        state = checkpoint["model_state_dict"]
        self.model.load_state_dict(state, strict=False)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    

        self.start_epoch = checkpoint["epoch"] + 1
        self.best_val_acc = checkpoint.get("best_val_acc", 0.0)
        self.epochs_no_improve = checkpoint.get("epochs_no_improve", 0)

        self.logger.info(f"Resuming training from epoch {self.start_epoch} with best_val_acc = {self.best_val_acc:.4f}")
        self.logger.info(f"Num of epochs where val acc has not improved: {self.epochs_no_improve}")

        self.model.train()

    def train(self):
        """
        Main training loop. Handles fresh training or resume, sets up dataloaders/model,
        logs progress, checkpoints, early stopping, and runs visualizations.
        """
        if self.resume_from_checkpoint:
            self.resume_training()
        else:
            self.setup_dataloaders()
            self.setup_model()
            # initialize for fresh run
            self.start_epoch = 1
            self.best_val_acc = 0.0
            self.epochs_no_improve = 0

        utils.print_model_size(self.model, only_trainable=True)
        utils.count_flops(self.model, input_size=(1, 3, 32, 32))

        # Setup visualization: watch model for gradients in wandb or sample images in TensorBoard
        if self.log_backend:
            self.setup_backend_visualization()

        self.start_time = time.time()
        self.logger.info("Starting training...")

        for epoch in range(self.start_epoch, self.epochs + 1):
            self.current_epoch = epoch

            self.model.train()

            train_running_loss = 0.0
            correct_samples = 0
            total_samples = 0

            # Batch Training
            num_batches = len(self.train_loader)
            for batch_idx, (image, labels) in enumerate(self.train_loader, start=1):
                print(f"[Epoch {epoch}] Batch {batch_idx}/{num_batches} — training…")
                image = image.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad() # zero out gradients before new backward pass

                if self.use_amp: # Use mixed-precision if arg is set
                    with autocast(str(self.device)):
                        outputs, attn_maps = self.model(image, return_attn=True)
                        loss = self.criterion(outputs, labels)
                    # scale, backward, step, update
                    self.scalar.scale(loss).backward()
                    self.scalar.step(self.optimizer)
                    self.scalar.update()
                else:
                    # Else use vanilla FP32
                    outputs, attn_maps = self.model(image, return_attn=True)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()

                train_running_loss += loss.item() * image.size(0) # running loss = train_running_loss + loss * batch_size. --.item() converts tensor to true value
                _, preds = outputs.max(dim=1)
                correct_samples += preds.eq(labels).sum().item() # correct_samples = correct_samples + (true(1), false(0), true(1), true(0)).sum()
                total_samples += labels.size(0) # total_samples = total_samples + labels batch_size

            train_loss = train_running_loss / total_samples
            train_acc = correct_samples / total_samples * 100.0

            # Validation phase
            self.model.eval()
            val_running_loss = 0.0
            val_total_samples = 0
            correct_val = 0
            val_num_batches = len(self.val_loader)

            with torch.no_grad():
                for batch_idx, (image, labels) in enumerate(self.val_loader, start=1):
                    print(f"[Epoch {epoch}] Batch {batch_idx}/{val_num_batches} — validating…")
                    image = image.to(self.device)
                    labels = labels.to(self.device)

                    if self.use_amp: # Use mixed-precision if arg is set
                        with autocast(str(self.device)):
                            outputs, attn_maps = self.model(image, return_attn=True)
                            loss = self.criterion(outputs, labels)
                    else: # Else use vanilla FP32
                        outputs, attn_maps = self.model(image, return_attn=True)
                        loss = self.criterion(outputs, labels)

                    val_running_loss += loss.item() * image.size(0) # running loss = val_running_loss + loss * batch_size. --.item() converts tensor to true value
                    _, preds = outputs.max(dim=1)
                    correct_val += preds.eq(labels).sum().item() # correct_val = correct_val + (true(1), false(0), true(1), true(0)).sum()
                    val_total_samples += labels.size(0) # val_total_samples = val_total_samples + labels batch_size

            val_loss = val_running_loss / val_total_samples
            val_acc = correct_val / val_total_samples * 100.0

            # Adjust learning rate based on validation loss
            self.scheduler.step()

            # Logging
            log_dict = {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc, "learning_rate": self.scheduler.get_last_lr()[0]}
            
            self.backend_vis.log_scalars(log_dict, step=epoch)

            if self.log_backend in ("tensorboard", "both"):
                sample_batch_tensor, _ = self.sample_batch
                unnormed = unnormalize(sample_batch_tensor[:16])
                self.backend_vis.log_images("Sample images", unnormed, step=epoch)

            self.logger.info(f'Epoch {epoch:>2}/{self.epochs:>2} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%, LR: {self.scheduler.get_last_lr()[0]:.6f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')

            if self.vit_visualizers:
                self.model.eval()
                self.output_model_visualizations(epoch)
                torch.cuda.empty_cache()

            if self.save_every_n_epochs > 0 and epoch % self.save_every_n_epochs == 0:
                self.logger.info(f"Saving checkpoint at epoch {epoch} (every {self.save_every_n_epochs} epochs")
                save_path = os.path.join(self.out_dir, f"vit_cifar100_ep{epoch}.pth")
                torch.save({"epoch": epoch, "model_state_dict": self.model.state_dict(), "optimizer_state_dict": self.optimizer.state_dict(),
                             "scheduler_state_dict": self.scheduler.state_dict(), "best_val_acc": self.best_val_acc,"epochs_no_improve": self.epochs_no_improve}, save_path)
                # Save out model (ViT) visualizations
                # for now, vit visualizers will only be added if the model improves to see how visualizers progess as model improves (if debugging, move this block)
                if self.vit_visualizers:
                    # Save visualizations as zip (allows visualizations to repopulate if resuming training from last checkpoint)
                    save_visualization_zip(self.visualizations_dir, epoch)
                else:
                    self.logger.info('Visualization archiving is skipped because vit_visualizers is empty.')

            # Check for best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.epochs_no_improve = 0
                # Save checkpoint
                self.logger.info('last val acc > best_val_acc.. saving model')
                save_path = os.path.join(self.out_dir, "vit_cifar100_best.pth")
                torch.save({"epoch": epoch, "model_state_dict": self.model.state_dict(), "optimizer_state_dict": self.optimizer.state_dict(),
                             "scheduler_state_dict": self.scheduler.state_dict(), "best_val_acc": self.best_val_acc,"epochs_no_improve": self.epochs_no_improve}, save_path)
            else:
                self.epochs_no_improve += 1
                torch.cuda.empty_cache()
            # Early stopping condition
            if self.epochs_no_improve >= self.early_stopping:
                self.logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        # Evaluate on test set after training completes
        self.logger.info('testing...')
        self.test()

    def test(self):
        """
        Evaluates model on the held-out test set, logs overall accuracy/loss, and records final runtime.
        """
        self.model.eval()
        test_loss = 0.0
        correct_samples = 0
        total_samples = 0

        # Compute predictions on test set
        with torch.no_grad():
            for image, labels in self.test_loader:
                image, labels = image.to(self.device), labels.to(self.device)
                outputs = self.model(image)
                loss = self.criterion(outputs, labels)

                test_loss += loss.item() * image.size(0)
                _, preds = outputs.max(dim=1)
                correct_samples += preds.eq(labels).sum().item()
                total_samples += labels.size(0)
        # Final test accuracy
        avg_test_loss = test_loss / total_samples
        test_acc = correct_samples / total_samples * 100.0

        # clean scalar logging
        step = getattr(self, "current_epoch", 0)
        # Log test metrics to backend
        self.backend_vis.log_scalars({"test_loss": avg_test_loss, "test_acc": test_acc}, step=step)


        self.logger.info(f"Final Test Loss: {avg_test_loss:.4f} | Test Acc: {test_acc:.2f}%")

        # Log total runtime
        total_time = time.time() - self.start_time
        self.logger.info(f"\n Total training time: {total_time:.2f} seconds on {self.device}")