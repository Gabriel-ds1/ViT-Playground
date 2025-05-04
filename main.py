import tyro
from configs.config import TrainViTConfig
from trainers.trainer_vit import TrainerViT

def main(cfg: TrainViTConfig):
    trainer = TrainerViT(cfg)
    trainer.train()
    if hasattr(trainer, "backend_vis"):
        trainer.backend_vis.close()

if __name__ == "__main__":
    tyro.cli(main)