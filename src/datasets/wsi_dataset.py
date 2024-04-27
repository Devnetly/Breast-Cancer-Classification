import dotenv
import os
from torch.utils.data import Dataset
from typing import Callable,Any,Optional

### ugly but necessary
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

class WSIDataset(Dataset):
    
    def __init__(self,
        wsi_path : str,
        patch_size : int = 224,
        transform : Optional[Callable] = None
    ) -> None:
        
        super().__init__()

        self.wsi_path = wsi_path
        self.transform = transform

        self.osr = open_slide(filename=wsi_path)
        self.tiles = DeepZoomGenerator(osr=self.osr, tile_size=patch_size,overlap=0,limit_bounds=False)

        self.width, self.height = self.tiles.level_tiles[self.tiles.level_count - 1]

    def __getitem__(self, index : int) -> tuple[Any, int, int]:

        h = index // self.width
        w = index % self.width

        tile = self.tiles.get_tile(level=self.tiles.level_count - 1, address=(w, h))

        if self.transform is not None:
            tile = self.transform(tile)

        return tile, w, h

    def __len__(self) -> int:
        return self.width * self.height