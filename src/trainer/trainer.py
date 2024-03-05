import torch
from tqdm.notebook import tqdm
from torch import nn
from torchmetrics import Metric
from typing import Self,Any,Optional

class Trainer:
    """
        a class that allows training models and recording a history
        of different scores & the loss over time.
    """

    def __init__(self,
        optimizer:Optional[torch.optim.Optimizer] = None,
        loss:Optional[torch.nn.Module] = None,
        scheduler : Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        metrics : Optional[dict[str, Metric]] = None,
        device : str =  'cpu',
    ):
        """
            The constructor of the Trainer class.

            - optimizer : the optimizer used to update the weights of type torch.optim.Optimizer.
            - loss : the loss function of type torch.nn.Module.
            - metrics : dictionary of metrics to compute each epoch.
            - device : the device you wish to train the model in.
        """
        self.optimizer = optimizer
        self.loss = loss
        self.scheduler = scheduler
        self.metrics = metrics if metrics is not None else {}
        self.history = self.init_history()
        self.model = None
        self.device = device

    def set_optimizer(self, optimizer : torch.optim.Optimizer) -> Self:
        """
            sets the optimizer to train the model on.
        
            - optimizer : torch.optim.Optimizer.

            Retuns : 
            - self.
        """
        self.optimizer = optimizer
        return self

    def set_loss(self, loss : torch.nn.Module) -> Self:
        """
            sets the loss of the model.
        
            - loss : torch.nn.Module.

            Retuns : 
            - self.
        """
        self.loss = loss
        return self

    def set_device(self, device : str) -> Self:
        """
            sets the device of the model.
        
            - device : either cpu or cuda.

            Retuns : 
            - self.
        """
        self.device = device
        return self

    def add_metric(self,name:str,metric:Metric) -> Self:
        """
            add a metric to the list of metrics already specified.

            - name : the name of metric,of type str.
            - metric : the metric to add,of type torchmetrics.Metric.

            Retuns : 
            - Self.
        """
        self.metrics[name] = metric
        self.history['train'][name] = []
        self.history['val'][name] = []
        return self
    
    def set_scheduler(self, scheduler : Optional[torch.optim.lr_scheduler.LRScheduler] = None) -> Self:
        """
            add a metric to the list of metrics already specified.

            - name : the name of metric,of type str.
            - metric : the metric to add,of type torchmetrics.Metric.

            Retuns : 
            - Self.
        """
        self.scheduler = scheduler
        return self

    def init_history(self) -> dict:
        """
            Returns :

            - a dictionary of arrays to store the history of the metrics. 
        """

        history = {
            'train' : {},
            'val' : {}
        }

        for name in list(self.metrics.keys()) + ['loss']:
            history['train'][name] = []
            history['val'][name] = []

        return history
    
    def reset(self) -> None:
        """
            resets the model by clearing the history.

            Retuns : 
            - None
        """
        self.history = self.init_history()

    def compute_metrics(self,
        y:torch.Tensor,
        y_hat:torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
            compute the different metrics.

            - y : torch.Tensor.
            - y_hat : torch.Tensor.

            Retuns : 
            - None.
        """

        result = {}

        for name,metric in self.metrics.items():
            result[name] = metric(y_hat, y).item()

        return result
    
    def add_dicts(self, dict1 : dict, dict2 : dict) -> dict[str,torch.Tensor]:
        """
            adds the values of tow dictionaries together.

            Returns : 
            - a new dictionary containing the results.
        """

        result = {}

        for key,value in dict1.items():
            result[key] = value + dict2[key]

        return result
    
    def div_dict(self, dict_ : dict, by : Any):
        """
            divides the values of dictionary by a given value.

            Returns : 
            - a new dictionary containing the results.
        """

        result = {}

        for key,value in dict_.items():
            result[key] = dict_[key] / by

        return result
    
    def get_results_dict(self) -> dict[str,float] :
        """
            Returns :
            - a new dictionary containing the keys of the metrics with a default value of zero.
        """
        
        results = {}

        for name in list(self.metrics.keys()) + ['loss']:
            results[name] = 0.0

        return results


    def append_to_history(self, results : dict[str,torch.Tensor], to : str = 'train') -> None:
        """
            appends a dictionary of results to the history.

            - results : dictionary of metrics.
            - to : whether to append to the validationsor the training history,must be string either 'train' ot 'val'            
        
            Retruns : 
            - None.
        """
        
        for key,value in results.items():
            self.history[to][key].append(value)


    def train_on_batch(self,X_batch : torch.Tensor,y_batch : torch.Tensor) -> tuple[float,torch.Tensor]:
        """
            applies one training step on a batch

            - X_batch : torch.Tensor.
            - y_batch : torch.Tensor.

            Returns :
            - float : the loss.
            - torch.Tensor : the predictions.
        """

        # put the data in the rightd device
        X_batch,y_batch = X_batch.to(self.device),y_batch.to(self.device)

        # 1- Forward pass
        y_hat = self.model(X_batch)

        # 2- Calculate the loss
        loss = self.loss(y_hat, y_batch)

        # 3- zero the gradients
        self.optimizer.zero_grad()

        # 4- Backward pass
        loss.backward()

        # 5- Optimizer step
        self.optimizer.step()

        return loss.item(),y_hat

    def test_on_batch(self,X_batch:torch.Tensor,y_batch:torch.Tensor) -> tuple[float,torch.Tensor]:
        """
            applies one validation step on a batch.

            - X_batch : torch.Tensor.
            - y_batch : torch.Tensor.

            Returns :
            - float : the loss.
            - torch.Tensor : the predictions.
        """

        # put the data in the rightd device
        X_batch,y_batch = X_batch.to(self.device),y_batch.to(self.device)

        # 1- Forward pass
        y_hat = self.model(X_batch)

        # 2- Calculate the loss
        loss = self.loss(y_hat, y_batch)

        return loss.item(),y_hat

    def check(self) -> None:

        if self.optimizer is None:
            raise Exception("optimizer can not be None.")
        
        if self.loss is None:
            raise Exception("loss can not be None.")

    def train(self,
        model:nn.Module,
        train_dataloader : torch.utils.data.DataLoader,
        val_dataloader : Optional[torch.utils.data.DataLoader] = None,
        epochs : int = 100
    ) -> None:
        """
            trains the model for number of epochs.

            - model : the model to train, of type torch.nn.Module.
            - train_dataloader : the loader to load the training batches,of type torch.utils.data.DataLoader.
            - val_dataloader : the loader to load the validation batches,of type torch.utils.data.DataLoader.
            - epochs : the number of epochs to train the model,of type int.

            Returns : 
            - None
        """

        # check if the parameters are valid.
        self.check()

        self.model = model

        for _ in range(epochs):

            # put the model in training mode
            self.model.train()

            train_results = self.get_results_dict()
            val_results = self.get_results_dict()

            ### Training loop
            for _,(X_batch,y_batch) in enumerate(tqdm(train_dataloader)):
                
                loss, y_hat = self.train_on_batch(X_batch,y_batch)
                train_batch_results = self.compute_metrics(y_batch,y_hat)
                train_batch_results['loss'] = loss
                train_results = self.add_dicts(train_results, train_batch_results)

            ### Testing loop
            if val_dataloader is not None:
                # put the model in training mode
                self.model.eval()

                with torch.inference_mode():
                    for X_batch,y_batch in val_dataloader:
                        loss, y_hat = self.test_on_batch(X_batch,y_batch)
                        val_batch_results = self.compute_metrics(y_batch,y_hat)
                        val_batch_results['loss'] = loss
                        val_results = self.add_dicts(val_results, val_batch_results)

            # adjust the metrics
            train_results = self.div_dict(train_results, by=len(train_dataloader))
            val_results = self.div_dict(val_results, by=len(val_dataloader))

            # append to the history
            self.append_to_history(train_results, to='train')
            self.append_to_history(val_results, to='val')

            # update the learning rate if a scheduler is defined
            if self.scheduler is not None:
                self.scheduler.step()