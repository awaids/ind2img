from PIL import Image
from pathlib import Path
from functools import cache
import numpy as np

class array2img:
    def __init__(self, data:np.ndarray) -> None:
        # Manage 2D arrays with capabilities to display
        assert(len(data.shape) == 2), "data shape must be 2D"
        self.data = data
    
    @cache
    def _get_Image(self) -> Image:
        return Image.fromarray(self.data, mode = 'L')

    def draw(self) -> None:
        self._get_Image().show()

    def save(self, png:Path) -> None:
        self._get_Image().save(png)



# s = 512
# w, h = s, s

# ones = np.ones(shape=(w,h), dtype=np.uint8) * 255
# zeros = np.zeros(shape=(w,h), dtype=np.uint8)

# red = np.random.randint(low=0, high=255, size=(w,h), dtype=np.uint8)
# green = np.random.randint(low=0, high=255, size=(w,h), dtype=np.uint8)
# blue = np.random.randint(low=0, high=255, size=(w,h), dtype=np.uint8)
# data = np.stack((red,green, blue), axis=2)

# # data = np.stack((ones, ones, ones), axis=2)
# print(data.shape)
# img = Image.fromarray(data, 'RGB')
# img.save('my.png')
# img.show()

# # Draw with b/w data
# bw_data = np.random.randint(low=0, high=255, size=(w, h), dtype=np.uint8)
# print(bw_data.shape)
# print(len(bw_data.shape))
# img = Image.fromarray(bw_data, 'L')
# img.show()
