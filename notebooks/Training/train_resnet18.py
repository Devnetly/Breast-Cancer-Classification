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
from src.models import ResNet18
from src.transforms import LabelMapper
from src.trainer import Trainer
from src.utils import history2df

class DEFAULTS:
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

    return parser.parse_args()

def load_envirement_variables() -> tuple[str, str, str]:

    PATCHES_DIR = dotenv.get_key(dotenv.find_dotenv(), "PATCHES_DIR")
    HISTORIES_DIR = dotenv.get_key(dotenv.find_dotenv(), "HISTORIES_DIR")
    MODELS_DIR = dotenv.get_key(dotenv.find_dotenv(), "MODELS_DIR")

    return PATCHES_DIR,HISTORIES_DIR,MODELS_DIR

def load_model(models_dir : str) -> ResNet18:

    model = None

    if len(os.listdir(os.path.join(models_dir, "resnet18"))) > 0:

        print("-- Loading the last model's weights ---")

        weights_path = os.path.join(
            models_dir,
            'resnet18',
            os.listdir(os.path.join(models_dir, "resnet18"))[-1]
        )

        model = ResNet18(n_classes=GLOBAL.NUM_CLASSES) \
            .to(GLOBAL.DEVICE) 
        
        model.load_state_dict(torch.load(weights_path))
    else:
        model = ResNet18(n_classes=GLOBAL.NUM_CLASSES) \
            .to(GLOBAL.DEVICE)
        
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
        "0_N":"benign", # 0 is the label for benign (BY)
        "1_PB":"benign", 
        "2_UHD":"benign",
        "3_FEA":"atypical",
        "4_ADH":"atypical", # 1 is the label for atypical (AT)
        "5_DCIS":"malignant",
        "6_DCIS":"malignant", # 2 is the label for malignant (MT)
    })

    print("-- Creating datasets ---")



    dataset = datasets.ImageFolder(
        root=os.path.join(PATCHES_DIR,"train"), 
        transform=transform, 
        target_transform=label_mapper
    )

    print(dataset.class_to_idx)

    model = load_model(MODELS_DIR)
    
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