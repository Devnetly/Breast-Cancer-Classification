import torch
import pandas as pd
import os
import time
import numpy as np
from torch import nn,optim
from torch.utils.data import DataLoader
from torchmetrics import Metric
from typing import Optional
from tqdm import tqdm

class Trainer:

    def __init__(self,
        model : nn.Module,
        loss_fn : nn.Module,
        optimizer : optim.Optimizer,
        checkpoints_folder : str,
        history_filename : str,
        scheduler : Optional[optim.lr_scheduler._LRScheduler] = None,
        metrics : Optional[dict[str,Metric]] = None,
        device : Optional[torch.device] = None,
        save_every : Optional[int] = None,
        save_on_best : bool = False,
        score_metric : str = 'loss',
        score_direction : str = 'min',
    ) -> None:
        
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics = metrics if metrics is not None else {}
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_every = save_every
        self.save_on_best = save_on_best if save_on_best is not None else False
        self.score_metric = score_metric if score_metric is not None else 'loss'
        self.score_direction = score_direction if score_direction is not None else 'min'
        self.checkpoints_folder = checkpoints_folder
        self.history_filename = history_filename
        self.start_epoch = 0

        self.history = self._load_history()
        self._load_checkpoint()
    
    def _load_history(self) -> pd.DataFrame:
        
        if os.path.exists(self.history_filename):
            return pd.read_csv(self.history_filename)
        else:
            return pd.DataFrame(columns=[
                'epoch',
                'time',
                'split',
                'loss'
            ] + list(self.metrics.keys()))
        
    def _update_history(self, row : pd.Series) -> None:
        self.history.loc[self.history.shape[0]] = row

    def _save_history(self) -> None:
        self.history.to_csv(self.history_filename,index=False)

    def _load_checkpoint(self) -> None:

        checkpoints = os.listdir(self.checkpoints_folder)
        checkpoints = sorted(checkpoints,key=lambda x : int(x.split('_')[-1].split('.')[0]))

        if len(checkpoints) == 0:
            return

        print(f"Loading checkpoint {checkpoints[-1]}")

        checkpoint = torch.load(os.path.join(self.checkpoints_folder,checkpoints[-1]))

        msg = self.model.load_state_dict(checkpoint['model_state_dict'])
        print(msg)

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.start_epoch = checkpoint['epoch'] + 1

    def _save_checkpoint(self,epoch : int) -> None:

        checkpoint = {
            'epoch' : epoch,
            'model_state_dict' : self.model.state_dict(),
            'optimizer_state_dict' : self.optimizer.state_dict(),
            'scheduler_state_dict' : self.scheduler.state_dict() if self.scheduler is not None else None,
        }

        torch.save(checkpoint,os.path.join(self.checkpoints_folder,f'checkpoint_{epoch}.pt'))

    def step(
        self,
        loader : DataLoader,
        is_training : bool,
        epoch : int,
        total_epochs : int
    ) -> None:  
        
        ### Set model to train/eval mode
        self.model.train(mode=is_training)

        ### Start time
        tic = time.time()
        iterator = tqdm(loader,desc=f"Epoch {epoch+1}/{total_epochs} - {'Train' if is_training else 'Val'}")

        Y = []
        Y_PRED = []
        running_loss = 0.0

        for x,y in iterator:

            postfix = {}

            ### Move data to device
            x,y = x.to(self.device),y.to(self.device)

            with torch.set_grad_enabled(is_training):

                ### Forward pass
                y_pred = self.model(x)

                ### Compute loss
                loss = self.loss_fn(y_pred,y)

                ### Store predictions
                Y.append(y)
                Y_PRED.append(y_pred)
                
                if is_training:

                    ### Zero gradients
                    self.optimizer.zero_grad()

                    ### Backward pass
                    loss.backward()

                    ### Update weights
                    self.optimizer.step()

                    ### Update scheduler
                    if self.scheduler is not None:
                        self.scheduler.step()
                        lr = self.scheduler.get_last_lr()[-1]
                        postfix['lr'] = lr

                ### Update progress bar
                postfix['loss'] = loss.item()
                running_loss += loss.item()

                for key,metric in self.metrics.items():
                    postfix[key] = metric(y_pred,y).item()
                
                iterator.set_postfix(postfix)

        ### End time
        toc = time.time()

        ### Compute metrics
        Y = torch.cat(Y,dim=0)
        Y_PRED = torch.cat(Y_PRED,dim=0)

        results = pd.Series(
            index=self.history.columns,
            data=np.zeros(len(self.history.columns)),
            dtype=object
        )

        results['epoch'] = epoch
        results['time'] = toc - tic
        results['split'] = 'train' if is_training else 'val'
        results['loss'] = running_loss / len(loader)

        for key,metric in self.metrics.items():
            results[key] = metric(Y_PRED,Y).item()

        ### Update history
        self._update_history(results)

        ### Display results
        msg = f"\nEpoch {epoch+1}/{total_epochs} - {'Train' if is_training else 'Val'} : "

        for key in list(self.metrics.keys()) + ['loss']:
            msg += f"{key} = {results[key]} |"

        print(msg,end='\n\n')

        del Y,Y_PRED

    def train(
        self,
        train_loader : DataLoader,
        val_loader : DataLoader,
        epochs : int
    ) -> None:
        
        ### Move to device
        self.model.to(self.device)
        self.loss_fn.to(self.device)

        for metric in self.metrics.values():
            metric.to(self.device)
        
        ### Training loop
        for epoch in range(self.start_epoch,epochs + self.start_epoch):
            
            ### Training phase
            self.step(train_loader,is_training=True,epoch=epoch - self.start_epoch,total_epochs=epochs)

            ### Validation phase
            self.step(val_loader,is_training=False,epoch=epoch - self.start_epoch,total_epochs=epochs)

            ### Checkpointing
            if (self.save_every is not None and (epoch+1) % self.save_every == 0) or (epoch == epochs + self.start_epoch - 1):
                self._save_checkpoint(epoch)
                print(f"Checkpoint saved at epoch {epoch+1}.")
            elif self.save_on_best:
                
                mask = self.history['split'] == 'val'
                value = self.history[mask][self.score_metric].agg(self.score_direction)
                best_epoch = self.history[mask][self.history[self.score_metric] == value].iloc[0]['epoch']
                
                if epoch == best_epoch:
                    self._save_checkpoint(epoch)
                    print(f"Best checkpoint saved at epoch {epoch+1}, with {self.score_metric} = {value}.")
                
        self._save_history()