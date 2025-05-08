import torch

class RescaledScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, base_scheduler, initial_lr, eta_min):
        self.base_scheduler = base_scheduler
        self.initial_lr = initial_lr
        self.eta_min = eta_min
        self.optimizer = base_scheduler.optimizer
        self.last_epoch = base_scheduler.last_epoch
        super().__init__(self.optimizer)

    def get_lr(self):
        base_lrs = self.base_scheduler.get_last_lr()
        # Rescale base LR from [0, initial_lr] → [eta_min, initial_lr]
        return [self.eta_min + (lr / self.initial_lr) * (self.initial_lr - self.eta_min) for lr in base_lrs]

    def step(self, epoch=None):
        self.base_scheduler.step(epoch)