import os
import torch
import sys
import argparse
from timm.data import resolve_model_data_config,create_transform
from torch import nn
from torchvision.models.resnet import ResNet
from torchvision.transforms import ToTensor,Resize,Compose,Lambda,Normalize
from tqdm import tqdm
sys.path.append('../..')
from src.models import ResNet,ResNet18,ResNet34,vit_small,VisionTransformerHIPT
from src.utils import load_model_from_folder
from src.datasets import WSIPatchedDataset
from torch.utils.data import DataLoader
from timm.models.vision_transformer import VisionTransformer

def create_transforms(model : nn.Module,patch_size : int = 224):

    if isinstance(model, ResNet):
        return Compose([
            Resize(size=(patch_size,patch_size)),
            ToTensor(),
            Lambda(lambda x : x[:-1])
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif  isinstance(model, VisionTransformer):
        data_config = resolve_model_data_config(model)
        transforms = create_transform(**data_config, is_training=False)
        transforms = Compose([
            Resize(size=(patch_size,patch_size)),
            transforms.transforms[2],
            Lambda(lambda x : x[:-1]),
            transforms.transforms[3]
        ])
        return transforms
    elif isinstance(model, VisionTransformerHIPT):
        return Compose([
            Resize(size=(patch_size,patch_size)),
            ToTensor(), 
            Lambda(lambda x : x[:-1]),
            Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])
    else:
        raise Exception(f"{model.__class__.__name__} is not supported.")

def transform_wsis(
    model : ResNet,
    root : str,
    patch_size : int,
    coords_folder : str,
    tensors_folder : str,
    metadata_file : str,
    max_wsis : int,
    device : str,
    batch_size : int,
    num_workers : int,
    prefetch_factor : int,
    train : bool,
    test : bool,
    val : bool
) -> None:
    
    if not train and not test and not val:
        raise Exception("Specify at least one folder to process.")
    
    processed_wsis = []

    with open(metadata_file, "r") as f:

        content = f.read()

        if len(content) != 0:
            processed_wsis = content.split(',')

    wsis_paths : list[str] = []

    split_map = {
        "train" : train,
        "test" : test,
        "val" : val
    }

        
    for split in os.listdir(root):
            
        if split_map[split]:

            split_path = os.path.join(root, split)

            for category in os.listdir(split_path):

                category_path = os.path.join(split_path, category)

                for sub_category in os.listdir(category_path):

                    sub_category_path = os.path.join(category_path, sub_category)

                    wsis = os.listdir(sub_category_path)
                    wsis = list(filter(lambda x : x not in processed_wsis, wsis))
                    wsis = list(map(lambda x : os.path.join(sub_category_path, x),wsis))
                    wsis_paths.extend(wsis)

    to_process = len(wsis_paths) if max_wsis is None else min(len(wsis_paths), max_wsis)
    wsis_paths = wsis_paths[:to_process]

    transform = create_transforms(model, patch_size=patch_size)

    if to_process == 0:
        print("\n --- No Whole slides images to process --- \n")
    else:
        print(f"\n --- Processing : {to_process}\n")

    model.eval()

    with torch.inference_mode():

        for i, wsi_path in enumerate(wsis_paths):

            basename = os.path.basename(wsi_path)

            coords_path = wsi_path.replace(root, coords_folder).replace(basename, '')
            coords_path = os.path.join(coords_path,'patches',os.path.splitext(basename)[0]) + os.path.extsep + 'h5'

            print(f"{i} - Processing {wsi_path} with coords {coords_path}: \n")

            dataset = WSIPatchedDataset(wsi_path=wsi_path,patch_size=patch_size,transform=transform, coords_path=coords_path)
            loader = DataLoader(dataset=dataset,batch_size=batch_size,num_workers=num_workers,prefetch_factor=prefetch_factor)
            vector = []

            for tiles, _, _ in tqdm(loader):
                tiles = tiles.to(device)
                vectors = model(tiles).to('cpu')
                vector.append(vectors)

            vector = torch.cat(vector)

            relative_path = wsi_path.replace(root, '')[1:]
            path_to_save = os.path.join(tensors_folder, relative_path)
            path_to_save,_ = os.path.splitext(path_to_save)
            path_to_save += f"{os.path.extsep}pth"

            print(f"\n --- Whole slide image was saved to path : {path_to_save} -- \n")

            pardir = os.path.dirname(path_to_save)

            processed_wsis.append(basename)

            if not os.path.exists(pardir):
                os.makedirs(pardir, exist_ok=True)

            torch.save(vector, f=path_to_save)

            del vector
        
    with open(metadata_file, "w") as f:
        f.write(','.join(processed_wsis))

def main(args):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = None

    if args.model == "resnet18":
        model = ResNet18(n_classes=3)
        load_model_from_folder(model=model, weights_folder=args.model_weights,verbose=True)
        model.resnet.fc = nn.Identity()
    elif args.model == "resnet34":
        model = ResNet34(n_classes=3)
        load_model_from_folder(model=model, weights_folder=args.model_weights,verbose=True)
        model.resnet.fc = nn.Identity()
    elif args.model == "vit":
        model = VisionTransformer(img_size=args.patch_size, patch_size=16, embed_dim=384, num_heads=6, num_classes=0)
        load_model_from_folder(model=model, weights_folder=args.model_weights,verbose=True)
    elif args.model == "hipt":
        model = vit_small()
        load_model_from_folder(model=model, weights_folder=args.model_weights,verbose=True)
    else:
        raise Exception(f"model {args.model} is not available.")
        
    model = model.to(device)
    
    transform_wsis(
        model=model,
        root=args.in_path,
        patch_size=args.patch_size,
        tensors_folder=args.out_path,
        metadata_file=args.metadata_path,
        max_wsis=args.n,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        train=args.train,
        test=args.test,
        val=args.val,
        coords_folder=args.coords_path
    )

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--coords-path', type=str, required=True)
    parser.add_argument('--in-path', type=str, required=True)
    parser.add_argument('--out-path', type=str, required=True)
    parser.add_argument('--train', type=bool, default=False)
    parser.add_argument('--test', type=bool, default=False)
    parser.add_argument('--val', type=bool, default=False)
    parser.add_argument('--patch-size', type=int, default=224)
    parser.add_argument('--model', type=str, default="resnet18")
    parser.add_argument('--model-weights', type=str, required=True)
    parser.add_argument('--metadata-path', type=str, required=True)
    parser.add_argument('--n', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--prefetch-factor', type=int, default=None)

    args = parser.parse_args()

    main(args)