from torch.optim.optimizer import Optimizer
from math import cos,pi

class CosineScheduler:
    
    def __init__(self, 
        optimizer : Optimizer, 
        lr : float,
        num_steps_per_epoch : int,
        warmup_epoch : int = 0,
        min_lr : float = 0,
        last_epoch : int = 0, 
        train_epoch : int = 100
    ) -> None:
        
        self.optimizer = optimizer
        self.warmup_epoch = warmup_epoch
        self.lr = lr
        self.num_steps_per_epoch = num_steps_per_epoch
        self.min_lr = min_lr
        self.epoch = last_epoch
        self.train_epoch = train_epoch
        self.lr = lr

    def step(self) -> None:
        
        if self.epoch < self.warmup_epoch:
            lr = (self.lr * self.epoch) / self.warmup_epoch
        else:
            lr = self.min_lr + (self.lr - self.min_lr) * 0.5 * (1.0 + cos(pi * (self.epoch - self.warmup_epoch) / (self.train_epoch - self.warmup_epoch)))

        for param_grp in self.optimizer.param_groups:
            if 'lr_scale' in param_grp:
                param_grp['lr'] = lr * param_grp['lr_scale']
            else:
                param_grp['lr'] = lr

        self.epoch += 1 / self.num_steps_per_epoch
        self.lr = lr

    def get_lr(self):
        return self.lr