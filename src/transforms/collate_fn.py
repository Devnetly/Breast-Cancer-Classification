import torch

def make_patches(
    img : torch.Tensor,
    patch_width : int,
    patch_height : int
) -> list[torch.Tensor]:
    """
        Arguments :

        - img : the image as a tensor of shape (c,h,w)
        - patch_width : the width of the patch a positive integer.
        - patch_height : the height of the patch a positive integer.

        Retuns : 
        - a list of tensors,each tensor is of shape (c,patch_width,patch_height) representing one patch
    """

    patches = img \
        .unfold(1,patch_width,patch_width) \
        .unfold(2,patch_height,patch_height) \
        .flatten(1,2) \
        .permute(1,0,2,3)

    patches = list(patches)

    return patches

def collate_fn(batch : list[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor]:
    """
        a collate functions that patches each image in the batch.

        Usage Example : 
        
        ```Python
            from transforms import collate_fn

            dataset  =datasets.ImageFolder(root="<your-path>",transform=transforms.ToTensor())
            
            dataloader = DataLoader(dataset=dataset,batch_size=32,collate_fn=collate_fn)
        ```

        Arguments : 
        - batch : the batch to process.

        Retuns:
        - a tensor of shape (patches_num,c,h,w).
        - a tensor of labels
    """
    
    new_x = []
    new_y = []
    
    for x, y in batch:
        patches = make_patches(x, 224, 224)
        new_x.extend(patches)
        new_y.extend([y for _ in range(len(patches))])

    new_x = torch.stack(new_x)
    new_y = torch.tensor(new_y)
    
    return new_x,new_y