import sys
sys.path.append('../..')
import os
import torch
import torchmetrics
import dotenv
import time
import argparse
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
from src.models import ResNet18, ResNet34
from src.transforms import LabelMapper
from src.trainer import Trainer
from src.utils import history2df

class DEFAULTS:
    MODEL = "resnet18"
    BATCH_SIZE = 1
    LEARNING_RATE = 0.001
    EPOCHS = 1

class GLOBAL:
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_CLASSES = 3

def get_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument("--batch-size", type=int,default=DEFAULTS.BATCH_SIZE)
    parser.add_argument("--epochs", type=int,default=DEFAULTS.EPOCHS)
    parser.add_argument("--learning-rate", type=float,default=DEFAULTS.LEARNING_RATE)
    parser.add_argument("--model-type", type=str,default=DEFAULTS.MODEL)

    return parser.parse_args()

def load_envirement_variables() -> tuple[str, str, str]:

    PATCHES_DIR = dotenv.get_key(dotenv.find_dotenv(), "PATCHES_DIR")
    HISTORIES_DIR = dotenv.get_key(dotenv.find_dotenv(), "HISTORIES_DIR")
    MODELS_DIR = dotenv.get_key(dotenv.find_dotenv(), "MODELS_DIR")

    return PATCHES_DIR,HISTORIES_DIR,MODELS_DIR

def load_model(models_dir: str, model_type: str) -> torch.nn.Module:
    model = None

    #In case we already have weights load them and continue training
    if len(os.listdir(os.path.join(models_dir, model_type))) > 0:
        print(f"-- Loading the last {model_type} model's weights ---")

        weights_path = os.path.join(
            models_dir,
            model_type,
            os.listdir(os.path.join(models_dir, model_type))[-1]
        )

        if model_type == "resnet18":
            model = ResNet18(n_classes=GLOBAL.NUM_CLASSES).to(GLOBAL.DEVICE)
        elif model_type == "resnet34":
            model = ResNet34(n_classes=GLOBAL.NUM_CLASSES).to(GLOBAL.DEVICE)

        model.load_state_dict(torch.load(weights_path))

    #If we don't have weights, create a new model
    else:
        if model_type == "resnet18":
            model = ResNet18(n_classes=GLOBAL.NUM_CLASSES).to(GLOBAL.DEVICE)
        elif model_type == "resnet34":
            model = ResNet34(n_classes=GLOBAL.NUM_CLASSES).to(GLOBAL.DEVICE)

    return model

def main(args):

    print("Training with: " + 'cuda' if torch.cuda.is_available() else 'cpu')

    print("-- Loading envirement variables ---")

    PATCHES_DIR,HISTORIES_DIR,MODELS_DIR = load_envirement_variables()

    print("-- Creating transforms ---")

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    label_mapper = LabelMapper({
        0:0, # 0 is the label for benign (BY)
        1:0, 
        2:0,
        3:1, # 1 is the label for atypical (AT)
        4:1,
        5:2, # 2 is the label for malignant (MT)
        6:2,
    })

    print("-- Creating datasets ---")



    dataset = datasets.ImageFolder(
        root=os.path.join(PATCHES_DIR,"train"), 
        transform=transform, 
        target_transform=label_mapper
    )

    model = load_model(MODELS_DIR, args.model_type)
    
    print("-- Creating the dataloader ---")

    dataloader = DataLoader(
        dataset=dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
    )

    val_dataset = datasets.ImageFolder(
        root=os.path.join(PATCHES_DIR,"val"), 
        transform=transform, 
        target_transform=label_mapper
    )

    val_dataloader = DataLoader(
        dataset=val_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
    )

    print("-- Defining loss,optimizer and metrics ---")

    loss  = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    accuracy = torchmetrics \
        .Accuracy(num_classes=GLOBAL.NUM_CLASSES, task='multiclass') \
        .to(GLOBAL.DEVICE)

    trainer = Trainer(
        optimizer=optimizer, 
        loss=loss, 
        device=GLOBAL.DEVICE, 
        metrics={'accuracy': accuracy}
    )

    print("--- Begin Training ---")

    trainer.train(
        model=model,
        train_dataloader=dataloader,
        val_dataloader=val_dataloader,
        epochs=args.epochs
    )

    print("--- End Training ---")

    print("--- Save the history and the weights ---")

    # Save the history
    history2df(trainer.history).to_csv(
        os.path.join(HISTORIES_DIR,"resnet18",f"{time.time()}.csv"), 
        index=False
    )

    # Save the model
    torch.save(model.state_dict(),os.path.join(MODELS_DIR,"resnet18",f"{time.time()}.pt"))

if __name__ == '__main__':

    args = get_arguments()

    print(f"--- Trying starting training with args {args} ---")

    main(args)