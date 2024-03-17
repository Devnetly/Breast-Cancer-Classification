import sys
sys.path.append('../..')
import os
import torch
import torchmetrics
import time
import logging
import logging.config
from torchvision import datasets
from torch.utils.data import DataLoader
from src.transforms import LabelMapper
from src.trainer import Trainer
from src.utils import history2df
from helpers import *

logging.basicConfig(level = logging.INFO, format=' %(name)s :: %(levelname)-8s :: %(message)s')

logger = logging

def main(args):

    logger.info(f"training will starts with device : {GLOBAL.DEVICE}")

    if GLOBAL.DEVICE == 'cpu':
        logging.warning("cuda was not detected using cpu instead.")

    logger.info("loading envirement variables")

    PATCHES_DIR,HISTORIES_DIR,MODELS_DIR = load_envirement_variables()

    histories_folder = os.path.join(HISTORIES_DIR,args.histories_folder)
    weights_folder = os.path.join(MODELS_DIR,args.weights_folder)

    if not os.path.exists(histories_folder):
        raise Exception(f'no such a folder {histories_folder}')
    
    if not os.path.exists(weights_folder):
        raise Exception(f'no such a folder {weights_folder}')

    logger.info("creating transforms")

    label_mapper = LabelMapper({
        0:0, # 0 is the label for benign (BY)
        1:0, 
        2:0,
        3:1, # 1 is the label for atypical (AT)
        4:1,
        5:2, # 2 is the label for malignant (MT)
        6:2,
    })

    logger.info('Loading the weights')

    model = load_model(MODELS_DIR, args.weights_folder)

    logger.info("creating datasets")

    train_transform, val_transform = create_transforms(args.data_augmentation)

    dataset = datasets.ImageFolder(
        root=os.path.join(PATCHES_DIR,"train"), 
        transform=train_transform, 
        target_transform=label_mapper,
    )

    sampler = create_sampler(args.sampler,dataset)
    
    logger.info("creating dataloaders")

    dataloader = DataLoader(
        dataset=dataset, 
        batch_size=args.batch_size, 
        sampler=sampler
    )

    val_dataset = datasets.ImageFolder(
        root=os.path.join(PATCHES_DIR,"val"), 
        transform=val_transform, 
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
        os.path.join(histories_folder,f"{t}.csv"), 
        index=False
    )

    # Save the model
    torch.save(model.state_dict(),os.path.join(weights_folder,f"{t}.pt"))

if __name__ == '__main__':

    args = get_arguments()

    print(f"--- Trying starting training with args {args} ---")

    main(args)