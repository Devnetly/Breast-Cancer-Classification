import os
import dotenv

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