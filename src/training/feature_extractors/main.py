import sys
sys.path.append('../../..')
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
from src.utils import history2df,load_envirement_variables
from helpers import *
from torch.optim.lr_scheduler import ExponentialLR

logging.basicConfig(level = logging.INFO, format=' %(name)s :: %(levelname)-8s :: %(message)s')

logger = logging

def main(args):

    logger.info(f"training will starts with device : {GLOBAL.DEVICE}")

    if GLOBAL.DEVICE == 'cpu':
        logging.warning("cuda was not detected using cpu instead.")

    logger.info("loading envirement variables")

    PATCHES_DIR,HISTORIES_DIR,MODELS_DIR,DATA_DIR = load_envirement_variables()

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

    model = load_model(
        os.path.join(MODELS_DIR, args.weights_folder), 
        args.model_type, 
        args.dropout,
        depth=args.depth
    )

    logger.info("creating datasets")

    train_transform, val_transform = create_transforms(
        args.model_type,
        args.preprocessing,
        template_img_src=os.path.join(DATA_DIR, "train", "0_N", "BRACS_280_N_1.png"),
        config_file=os.path.join('.', 'BRACS.yaml')
    )

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
        sampler=sampler,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor
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
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor
    )

    logger.info("creating optimizer,loss and trainer instances")

    loss  = torch.nn.CrossEntropyLoss(weight=get_class_weights(dataset=dataset, class_weights=args.class_weights)).to(GLOBAL.DEVICE)

    optimizer = create_optimizer(
        model.parameters(),
        type=args.optimizer,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        momentum=args.momentum
    )

    scheduler = ExponentialLR(optimizer=optimizer,gamma=args.decay_rate,last_epoch=args.last_epoch)

    accuracy = torchmetrics \
        .Accuracy(num_classes=GLOBAL.NUM_CLASSES, task='multiclass') \
        .to(GLOBAL.DEVICE)
    
    f1_score = torchmetrics \
        .F1Score(num_classes=GLOBAL.NUM_CLASSES,task='multiclass',average='macro') \
        .to(GLOBAL.DEVICE)

    trainer = Trainer() \
        .set_optimizer(optimizer=optimizer) \
        .set_loss(loss=loss) \
        .set_scheduler(scheduler=scheduler) \
        .set_device(device=GLOBAL.DEVICE) \
        .add_metric("accuracy", accuracy) \
        .add_metric("f1_score", f1_score) \
        .set_save_best_weights(True) \
        .set_score_metric("f1_score")

    logger.info("training starts now")

    trainer.train(
        model=model,
        train_dataloader=dataloader,
        val_dataloader=val_dataloader,
        epochs=args.epochs
    )

    logger.info("training has ended")

    logger.info(f"best epoch = {trainer.best_epoch+1}")

    logger.info("saving results to the disk")

    # Save the history
    t = time.time()

    history2df(trainer.history).to_csv(
        os.path.join(histories_folder,f"{t}.csv"), 
        index=False
    )

    # Save the model
    torch.save(model.state_dict(),os.path.join(weights_folder,f"{t}.pt"))
    torch.save(trainer.best_weights,os.path.join(weights_folder,f"{t}_best_weights.pt"))

if __name__ == '__main__':

    args = get_arguments()

    print(f"--- Trying starting training with args {args} ---")

    main(args)