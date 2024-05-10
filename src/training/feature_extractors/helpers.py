import torch
import argparse
import sys
sys.path.append("../../..")
from torch.utils.data import RandomSampler,Sampler
from torchvision import transforms,datasets
from src.models import ResNet18,ResNet34,ResNet50
from src.utils import load_model_from_folder
from src.transforms import ReinhardNotmalizer
from torchsampler import ImbalancedDatasetSampler
from torch.optim import SGD,Adam,Optimizer,RMSprop
from randstainna.randstainna import RandStainNA
from torchvision.datasets import ImageFolder
from timm.data import resolve_model_data_config,create_transform
from timm.models import VisionTransformer
from torch import nn

class DEFAULTS:
    MODEL = "resnet18"
    BATCH_SIZE = 1
    LEARNING_RATE = 0.001
    EPOCHS = 1
    SAMPLER = "random"
    PREPROCESSING = "nothing"
    DROPOUT = 0.0
    DECAY_RATE = 1.0
    OPTIMIZER = "adam"
    LAST_EPOCH = -1
    LOSS = "ce"
    WEIGHT_DECAY = 1e-3
    DEPTH = 2
    NUM_WORKERS = 0
    PREFETCH_FACTOR = None
    CLASS_WEIGHTS = None
    MOMENTUM = 0.0

class GLOBAL:
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_CLASSES = 3
    PATCH_SIZE = 224

def get_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument("--batch-size", type=int,default=DEFAULTS.BATCH_SIZE)
    parser.add_argument("--epochs", type=int,default=DEFAULTS.EPOCHS)
    parser.add_argument("--learning-rate", type=float,default=DEFAULTS.LEARNING_RATE)
    parser.add_argument("--model-type", type=str,default=DEFAULTS.MODEL,choices=["resnet18","resnet34","resnet50","vit"])
    parser.add_argument("--weights-folder", type=str, required=True)
    parser.add_argument("--histories-folder", type=str, required=True)

    parser.add_argument("--preprocessing", type=str, default=DEFAULTS.PREPROCESSING, choices=[
        'nothing',
        'stain-normalization',
        'augmentation',
        'stain-augmentation'
    ])

    parser.add_argument("--sampler", type=str,default=DEFAULTS.SAMPLER, choices=["random","balanced"])
    parser.add_argument('--dropout', type=float,default=DEFAULTS.DROPOUT)
    parser.add_argument('--decay-rate', type=float, default=DEFAULTS.DECAY_RATE)
    parser.add_argument('--optimizer', type=str, default=DEFAULTS.OPTIMIZER, choices=["adam", "sgd", "rmsprop"])
    parser.add_argument('--last-epoch', type=int, default=DEFAULTS.LAST_EPOCH)
    parser.add_argument('--weight-decay', type=float, default=DEFAULTS.WEIGHT_DECAY)
    parser.add_argument('--depth', type=int, default=DEFAULTS.DEPTH)
    parser.add_argument('--num-workers', type=int, default=DEFAULTS.NUM_WORKERS)
    parser.add_argument('--prefetch-factor', type=int, default=DEFAULTS.PREFETCH_FACTOR)
    parser.add_argument("--class-weights", type=float, default=DEFAULTS.CLASS_WEIGHTS)
    parser.add_argument("--momentum", type=float, default=DEFAULTS.MOMENTUM)

    return parser.parse_args()

def load_model(
    models_dir: str, 
    model_type: str,
    dropout_rate : float,
    depth : int
) -> torch.nn.Module:

    model = None

    if model_type == "resnet18":
        model = ResNet18(n_classes=GLOBAL.NUM_CLASSES,dropout_rate=dropout_rate,depth=depth)
    elif model_type == "resnet34":
        model = ResNet34(n_classes=GLOBAL.NUM_CLASSES,dropout_rate=dropout_rate,depth=depth)
    elif model_type == "resnet50":
        model = ResNet50(n_classes=GLOBAL.NUM_CLASSES,dropout_rate=dropout_rate,depth=depth)
    elif model_type == "vit":
        model = VisionTransformer(
            img_size=GLOBAL.PATCH_SIZE, 
            patch_size=16, 
            embed_dim=384, 
            num_heads=6, 
            num_classes=GLOBAL.NUM_CLASSES, 
            drop_rate=dropout_rate,
            pos_drop_rate = dropout_rate,
            patch_drop_rate = dropout_rate,
            proj_drop_rate = dropout_rate,
            attn_drop_rate = dropout_rate,
            drop_path_rate = dropout_rate,
        )
    else:
        raise Exception(f'model {model_type} is not supported.')
    
    model = model.to(GLOBAL.DEVICE)

    load_model_from_folder(
        model=model, 
        weights_folder=models_dir,
        verbose=True,
    )

    return model

def create_sampler(
    type_ : str,
    dataset : datasets.ImageFolder
) -> Sampler:

    sampler = None

    if type_ == "random":
        sampler = RandomSampler(data_source=dataset)
    elif type_ == "balanced":
        sampler = ImbalancedDatasetSampler(dataset=dataset)
    else:
        raise Exception(f"sampler {type_} is not supported.")
    
    return sampler

def get_model_transforms(model : nn.Module):

    if isinstance(model, ResNet18) or isinstance(model, ResNet34) or isinstance(model, ResNet50):
        return [
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    elif isinstance(model, VisionTransformer):
        return [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    else:
        raise Exception(f"{model} is not supported.")

def create_transforms(
    model : nn.Module,
    type : str,
    template_img_src : str,
    config_file : str
) -> tuple[transforms.Compose,transforms.Compose]:

    train_transforms = []   
    val_transforms = []

    basic_train_transforms = get_model_transforms(model)
    basic_val_transforms = get_model_transforms(model)

    if type == "nothing":

        train_transforms = basic_train_transforms
        val_transforms = basic_val_transforms

    elif type == "stain-normalization":

        train_transforms = [
            ReinhardNotmalizer(template_img_src=template_img_src),
            *basic_train_transforms
        ]

        val_transforms = [
            ReinhardNotmalizer(template_img_src=template_img_src),
            *basic_val_transforms
        ]
        
    elif type == "augmentation":

        train_transforms = [
            transforms.Pad(10, padding_mode='reflect'),
            transforms.RandomRotation(20),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            *basic_train_transforms
        ]

        val_transforms = [
            *basic_val_transforms
        ]
        
    elif type == "stain-augmentation":

        train_transforms = [
            RandStainNA(
                yaml_file=config_file,
                std_hyper=-0.3, 
                probability=1.0,
                distribution='normal', 
                is_train=True
            ),
            *basic_train_transforms
        ]

        val_transforms = [
            *basic_val_transforms
        ]

    else:
        raise Exception(f"{type} is not supported.")

    train_transform = transforms.Compose(train_transforms)
    val_transform = transforms.Compose(val_transforms)

    return train_transform,val_transform

def create_optimizer(
    params,
    type : str, 
    lr : float,
    weight_decay : float,
    momentum : float
) -> Optimizer:

    if type == "adam":
        return Adam(params, lr=lr, weight_decay=weight_decay)
    elif type == "sgd":
        return SGD(params, lr=lr, weight_decay=weight_decay, momentum=momentum)
    elif type == "rmsprop":
        return RMSprop(params, lr=lr, weight_decay=weight_decay, momentum=momentum)
    else:
        raise Exception(f"optimizer {type} not supported.")
    
def get_class_weights(dataset : ImageFolder, class_weights : float | None) -> torch.Tensor | None:

    if class_weights is None:
        return None
    else:

        weights = torch.zeros(len(set([dataset.target_transform(class_) for class_ in dataset.class_to_idx.values()]))).type(torch.float)

        for _, label in dataset.imgs:
            weights[dataset.target_transform(label)] += 1.0

        weights = torch.float_power(torch.divide(torch.scalar_tensor(1.0), weights), torch.scalar_tensor(class_weights))
        weights = weights / weights.sum()
        weights = weights.type(torch.float)

        print(f"weights = {weights}")

        return weights