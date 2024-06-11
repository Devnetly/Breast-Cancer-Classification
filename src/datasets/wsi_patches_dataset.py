import os
import dotenv
import sys
sys.path.append('../..')
from src.utils import get_coords
from torch.utils.data import Dataset
from typing import Callable,Any,Optional

try:
    OPENSLIDE_PATH = dotenv.get_key(dotenv.find_dotenv(), "OPENSLIDE_PATH")
except Exception as e:
    print("Error setting OpenSlide path:", str(e))


if hasattr(os, 'add_dll_directory'):
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
        from openslide import open_slide
        from openslide.deepzoom import DeepZoomGenerator
else:
    import openslide
    from openslide import open_slide
    from openslide.deepzoom import DeepZoomGenerator

class WSIPatchedDataset(Dataset):

    def __init__(self,
        wsi_path : str,
        coords_path : str,
        transform : Optional[Callable] = None,
        patch_size : int = 224
    ) -> None:
        
        super().__init__()

        self.wsi_path = wsi_path
        self.coords_path = coords_path
        self.transform = transform
        self.patch_size = patch_size

        self.osr = open_slide(filename=wsi_path)
        self.coords = get_coords(self.coords_path)

        self.tiles = DeepZoomGenerator(osr=self.osr, tile_size=patch_size,overlap=0,limit_bounds=False)

        self.width, self.height = self.tiles.level_tiles[self.tiles.level_count - 1]

    def __getitem__(self, index : int) -> Any:
        
        x, y = self.coords[index]
        tile = self.osr.read_region(location=(x, y), level=0, size=(self.patch_size,self.patch_size))

        if self.transform is not None:
            tile = self.transform(tile)

        return tile, x, y
    
    def __len__(self) -> int:
        return len(self.coords)