import numpy as np
from pathlib import Path
from array2img import array2img

def test_basic_bw():
    s = 512
    w, h = s, s
    bw_data = np.random.randint(low=0, high=255, size=(w, h), dtype=np.uint8)
    a2i = array2img(data=bw_data)
    png = Path(__file__).parent / 'test.png'
    a2i.save(png)
    assert(png.exists()), "PNG file not found"
    png.unlink()
