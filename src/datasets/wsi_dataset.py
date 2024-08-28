import sys
sys.path.append('../..')
from torch.utils.data import Dataset
from typing import Callable,Any,Optional
from src.utils.import_openslide import DeepZoomGenerator,open_slide
from src.utils import get_coords

class WSIDataset(Dataset):
    
    def __init__(self,
        wsi_path : str,
        patch_size : int,
        coords_path : Optional[str] = None,
        transform : Optional[Callable] = None
    ) -> None:
        
        super().__init__()

        self.wsi_path = wsi_path
        self.transform = transform
        self.coords_path = coords_path
        self.patch_size = patch_size

        self.osr = open_slide(filename=wsi_path)
        self.tiles = DeepZoomGenerator(osr=self.osr, tile_size=patch_size,overlap=0,limit_bounds=False)
        self.coords = get_coords(self.coords_path) if self.coords_path is not None else None

        self.width, self.height = self.tiles.level_tiles[self.tiles.level_count - 1]

    def __getitem__(self, index : int) -> tuple[Any, int, int]:

        if self.coords is None:
            x = (index % self.width) * self.patch_size
            y = (index // self.width) * self.patch_size
        else:
            x,y = self.coords[index]

        tile = self.osr.read_region(location=(x, y), level=0, size=(self.patch_size,self.patch_size)).convert("RGB")

        if self.transform is not None:
            tile = self.transform(tile)

        return tile, x, y

    def __len__(self) -> int:
        
        if self.coords is None:
            return self.width * self.height
        else:
            return len(self.coords)