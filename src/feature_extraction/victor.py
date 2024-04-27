import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, Resize, Compose
from torchvision.transforms import ToTensor, Resize

import os
import sys
import logging
import argparse
sys.path.append('../..')

from src.utils import get_coords
from src.models import ResNet, ResNet18, ResNet34
from src.datasets.image_directory import ImageDirectory

from openslide import OpenSlide
from tqdm import tqdm
from PIL import Image, PngImagePlugin

PngImagePlugin.MAX_TEXT_CHUNK = 128 * (1024 ** 2)


def setup_logger(logfile_name):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(logfile_name)
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger


class WSIPatchesDataset(Dataset):
    def __init__(self, wsi_path, coords_path, model, patch_size=224, transform=None):
        self.wsi_path = wsi_path
        self.coords = get_coords(coords_path)
        self.model = model
        self.transform = transform
        self.patch_size = patch_size

    def __getitem__(self, idx):
        wsi = OpenSlide(self.wsi_path)
        x, y = self.coords[idx]
        size = self.patch_size
        patch = wsi.read_region(location=(x, y), level=0, size=(size, size))

        if patch.mode != 'RGB':
            patch = patch.convert('RGB')

        if self.transform:
            patch = self.transform(patch)
        else:
            patch = ToTensor()(patch)

        patch = patch.unsqueeze(0)

        with torch.inference_mode():
            features = self.model(patch)

        wsi.close()

        return features, f"patch_{idx}.png"

    def __len__(self):
        return len(self.coords)


def process_one(wsi_path, output_dir, coords_path, model_name, model_weights, prefetch_factor, num_workers, patch_size=224, verbose=False,
                logger=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_name == "ResNet18":
        model = ResNet18(n_classes=3).to(device)
    elif model_name == "ResNet34":
        model = ResNet34(n_classes=3).to(device)
    else:
        raise Exception("Model not supported.")

    state_dict = torch.load(model_weights, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    model.resnet.fc = nn.Identity()

    transform = Compose([
        Resize(size=(patch_size, patch_size)),
        ToTensor()
    ])

    dataset = WSIPatchesDataset(wsi_path, coords_path, model, patch_size=224, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, prefetch_factor=prefetch_factor, num_workers=num_workers)

    vectors = []

    with torch.inference_mode():
        for batch_id, (features, paths) in tqdm(enumerate(dataloader), total=len(dataloader)):
            for feature, path in zip(features, paths):
                vectors.append(feature)

                if verbose:
                    print(f"Feature vector for {path} of {wsi_path} done.")

    name = os.path.join(output_dir, os.path.splitext(os.path.basename(wsi_path))[0] + ".pt")
    torch.save(torch.stack(vectors), name)

    if logger:
        logger.info(f"Finished processing WSI: {wsi_path}")


def process_all(args):
    input_dir = args.input_dir
    train = args.train
    val = args.val
    test = args.test
    patch_size = args.patch_size
    output_dir = args.output_dir
    model_name = args.model_name
    model_weights = args.model_weights
    coords_dir = args.coords_dir
    max_wsis = args.max_wsis
    prefetch_factor = args.prefetch_factor
    num_workers = args.num_workers
    verbose = args.verbose

    wsi_logger = setup_logger("wsi.log")

    if not train and not test and not val:
        raise Exception("Specify at least one folder to process.")

    wsis_paths = []
    processed_wsis = set()

    try:
        with open("wsi.log", "r") as f:
            for line in f:
                if "Finished processing WSI:" in line:
                    processed_wsis.add(line.split(":")[-1].strip())
    except FileNotFoundError:
        pass

    split_map = {
        "train": train,
        "test": test,
        "val": val
    }

    for split in os.listdir(input_dir):

        if split_map[split]:

            split_path = os.path.join(input_dir, split)

            for category in os.listdir(split_path):
                category_path = os.path.join(split_path, category)

                for sub_category in os.listdir(category_path):
                    sub_category_path = os.path.join(category_path, sub_category)

                    wsis = os.listdir(sub_category_path)
                    wsis = list(filter(lambda x: x not in processed_wsis, wsis))
                    wsis = list(map(lambda x: os.path.join(sub_category_path, x), wsis))
                    wsis_paths.extend(wsis)

    wsis_paths = wsis_paths[:min(len(wsis), max_wsis)]

    if len(wsis_paths) == 0:
        print("\n --- No Whole slides images to process --- \n")
        return

    with torch.inference_mode():
        for wsi_path in wsis_paths:
            basename = os.path.splitext(os.path.basename(wsi_path))[0]
            coords_path = os.path.join(coords_dir, "patches", f"{basename}.h5")
            process_one(wsi_path, output_dir, coords_path, model_name, model_weights, prefetch_factor, num_workers, patch_size=patch_size, verbose=verbose, logger=wsi_logger)


if __name__ == '__main__':

    # example args
    #     input-dir = "dataset/original/wsi",
    #     train = True,
    #     val = False,
    #     test = False,
    #     patch_size = 224,
    #     output-dir = "vectors",
    #     model-name = "ResNet18",
    #     model-weights = "models/1711552006.2607753.pt",
    #     coords_dir = "dataset/patched",
    #     max_wsis = 1,
    #     verbose = True

    parser = argparse.ArgumentParser()

    parser.add_argument('--input-dir', type=str, required=True)
    parser.add_argument('--train', type=bool, default=False)
    parser.add_argument('--test', type=bool, default=False)
    parser.add_argument('--val', type=bool, default=False)
    parser.add_argument('--patch-size', type=int, default=224)
    parser.add_argument('--output-dir', type=str, default="vectors")
    parser.add_argument('--model-name', type=str, default="ResNet18")
    parser.add_argument('--model-weights', type=str, required=True)
    parser.add_argument('--coords-dir', type=str, required=True)
    parser.add_argument('--max-wsis', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--prefetch-factor', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--verbose', type=lambda x : x.lower() == "true", default=True)

    args = parser.parse_args()

    process_all(args)
