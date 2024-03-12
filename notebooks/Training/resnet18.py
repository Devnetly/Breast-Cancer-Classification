import sys
sys.path.append('../..')
import os
import torch
import torchmetrics
import dotenv
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
from src.models import ResNet18
from src.transforms import ImageResizer,LabelMapper,collate_fn
from src.trainer import Trainer

class DEFAULTS:
    BATCH_SIZE = 1
    LEARNING_RATE = 0.001
    EPOCHS = 1

class GLOBAL:
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_CLASSES = 3

def main():

    transform = transforms.Compose([
        ImageResizer(),
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

    dataset = datasets.ImageFolder(
        root=os.path.join(dotenv.get_key(dotenv.find_dotenv(), "ROI_LATEST"),"trash"), 
        transform=transform, 
        target_transform=label_mapper
    )

    model = ResNet18(n_classes=GLOBAL.NUM_CLASSES).to(GLOBAL.DEVICE)

    dataloader = DataLoader(
        dataset=dataset, 
        batch_size=DEFAULTS.BATCH_SIZE, 
        shuffle=True,
        collate_fn=collate_fn
    )

    loss  = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=DEFAULTS.LEARNING_RATE)

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
        epochs=DEFAULTS.EPOCHS
    )

    print(trainer.history)

    print("--- End Training ---")

if __name__ == '__main__':
    main()