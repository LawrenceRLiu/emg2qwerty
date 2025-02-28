#custom trainer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from typing import Any, Dict, List, Tuple, Union, Optional, Callable, Literal
import wandb
import emg2qwerty.models as models


class Trainer_SingleGPU:
    """Basic custom trainer for single GPU training."""
    model: models.ParentModel
    criterion: callable
    optimizer: optim.Optimizer
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    use_wandb: bool


    def __init__(self, model_name: Literal["TDSConvCTCModule"],
                 model_kwargs: Dict[str, Any],
                 criterion: Callable,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: DataLoader,
                 optimizer_name: Literal["Adam"] = "Adam",
                 optimizer_kwargs: Dict[str, Any] = {"lr": 1e-3},
                 device: str = "cuda",
                 metrics: Optional[List[Callable]] = None,
                 use_wandb: bool = False,
                 watch_model: bool = False,
                 run_name: Optional[str] = None,
                 save_dir: str = "models") -> None:
                    
        """Initialize the trainer.

        Args:
            model_name (Literal[&quot;TDSConvCTCModule&quot;]): the name of the model class to instantiate.
            model_kwargs (Dict[str, Any]): the keyword arguments to pass to the model class constructor
            criterion (Callable): the callable criterion to use for training.
            train_loader (DataLoader): the training data loader.
            val_loader (DataLoader): the validation data loader.
            test_loader (DataLoader): the test data loader.
            optimizer_name (Literal[&quot;Adam&quot;], optional): the name of the optimizer to use. Defaults to Adam.
            optimizer_kwargs (_type_, optional): the keyword arguments to pass to the optimizer constructor, Defaults to {"lr": 1e-3}.
            device (str, optional): the device to use for training. Defaults to "cuda".
            metrics (Optional[List[Callable]], optional): the metrics to use. Defaults to None. Expected that each metric will return a dictionary of metrics.
            use_wandb (bool, optional): whether to use wandb. Defaults to False.
            watch_model (bool, optional): whether to watch the model with wandb. Defaults to False.
            run_name (Optional[str], optional): the name of the wandb run. Defaults to None, in which case uses the default random wandb run name if use_wandb is True, else counts the number of runs in save_dir and names it "run_{count}".
            save_dir (str, optional): the directory to save the model. Defaults to "models".
        """
        #TODO: Implement the __init__ method
        pass 

    def create_model_and_optimizer_(self, model_name: Literal["TDSConvCTCModule"], model_kwargs: Dict[str, Any], optimizer_name: Literal["Adam"], optimizer_kwargs: Dict[str, Any])-> None:
        """Creates the model and optimizer.

        Args:
            model_name (Literal[&quot;TDSConvCTCModule&quot;]): the name of the model class to instantiate.
            model_kwargs (Dict[str, Any]): the keyword arguments to pass to the model class constructor.
            optimizer_name (Literal[&quot;Adam&quot;]): the name of the optimizer to use.
            optimizer_kwargs (Dict[str, Any]): the keyword arguments to pass to the optimizer constructor.

        """
        #TODO: Implement the create_model_and_optimizer_ method

        pass


    def train_one_epoch(self) -> Dict[str, float]:
        """Train the model for one epoch.

        Returns:
            Dict[str, float]: the training metrics.
        """
        #TODO: Implement the train_one_epoch method
        pass

    def train(self, num_epochs: int, val_interval: int = 1) -> None:
        """Train the model.

        Args:
            num_epochs (int): the number of epochs to train for.
            val_interval (int, optional): the validation interval. Defaults to 1.
        """

        #TODO: Implement the train method
        pass

    def evaluate(self, loader:Literal["val", "test"]) -> Tuple[float, Dict[str, float]]:
        """Evaluate the model.

        Args:
            loader (Literal[&quot;val&quot;, &quot;test&quot;]): the data loader to use.

        Returns:
            Tuple[float, Dict[str, float]]: the loss and the metrics.
        """
        #TODO: Implement the evaluate method
        pass

    def save_as_state_dict(self, path: str,
                           save_optim: bool = False) -> None:
        """Save the model as a state dict.

        Args:
            path (str): the path to save the model to.
            save_optim (bool, optional): whether to save the optimizer state. Defaults to False.
        """
        #TODO: Implement the save_as_state_dict method
        pass

    def load_from_state_dict(self, path: str,
                                load_optim: bool = False) -> None:
        """Load the model from a state dict.

        Args:
            path (str): the path to load the model from.
            load_optim (bool, optional): whether to load the optimizer state. Defaults to False.
        """
        #TODO: Implement the load_from_state_dict method
        pass






