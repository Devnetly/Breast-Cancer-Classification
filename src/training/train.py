import sys
sys.path.append('../..')
import os
import torch
import torchmetrics
import dotenv
import time
import argparse
import logging
import logging.config
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
from src.models import ResNet18, ResNet34
from src.transforms import LabelMapper
from src.trainer import Trainer
from src.utils import history2df,load_model_from_folder

logging.basicConfig(level = logging.INFO, format=' %(name)s :: %(levelname)-8s :: %(message)s')

logger = logging


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

    if model_type == "resnet18":
        model = ResNet18(n_classes=GLOBAL.NUM_CLASSES).to(GLOBAL.DEVICE)
    elif model_type == "resnet34":
        model = ResNet34(n_classes=GLOBAL.NUM_CLASSES).to(GLOBAL.DEVICE)

    load_model_from_folder(
        model=model, 
        weights_folder=os.path.join(models_dir, model_type),
        verbose=True,
    )

    return model

def main(args):

    logger.info(f"training will starts with device : {GLOBAL.DEVICE}")

    if GLOBAL.DEVICE == 'cpu':
        logging.warning("cuda was not detected using cpu instead.")

    logger.info("loading envirement variables")

    PATCHES_DIR,HISTORIES_DIR,MODELS_DIR = load_envirement_variables()

    logger.info("creating transforms")

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

    logger.info("creating datasets")


    dataset = datasets.ImageFolder(
        root=os.path.join(PATCHES_DIR,"train"), 
        transform=transform, 
        target_transform=label_mapper
    )

    model = load_model(MODELS_DIR, args.model_type)
    
    logger.info("creating dataloaders")

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

    logger.info("creating optimizer,loss and trainer instances")

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

    logger.info("training starts now")

    trainer.train(
        model=model,
        train_dataloader=dataloader,
        val_dataloader=val_dataloader,
        epochs=args.epochs
    )

    logger.info("training has ended")

    logger.info("saving results to the disk")

    # Save the history
    t = time.time()

    history2df(trainer.history).to_csv(
        os.path.join(HISTORIES_DIR,args.model_type,f"{t}.csv"), 
        index=False
    )

    # Save the model
    torch.save(model.state_dict(),os.path.join(MODELS_DIR,args.model_type,f"{t}.pt"))

if __name__ == '__main__':

    args = get_arguments()

    print(f"--- Trying starting training with args {args} ---")

    main(args)