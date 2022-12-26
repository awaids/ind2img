import math
import numpy as np
import pandas as pd
from PIL import Image

from pathlib import Path
from typing import List
from sklearn.preprocessing import MinMaxScaler

class Df2ImageBatchNormalizer:
    """ This normalizer normalizes for images but does batch normalization """
    def __init__(self, batch:int) -> None:
        assert(batch >= 3), "batch must be >=3 else you will get binary values"
        self.batch = batch
        self.scaler = MinMaxScaler(feature_range=(0, 255)) 
    
    def normalize(self, df: pd.DataFrame) -> np.ndarray:
        """ Normalizes the df based on the batch size """
        n_rows = df.shape[0]
        assert(n_rows >= self.batch), "Cannot batch-normalize, not enough rows in df"
        # Normaliztation technique: Normalize a sliced df, and then pick the last entry!
        new_arr = [self.scaler.fit_transform(df[df.columns].iloc[i:self.batch + i])[-1] for i in range(n_rows - self.batch + 1)]
        return np.array(new_arr).astype(np.uint8)

class df2array:
    """ Class to covert df into an ndarray where all rows are converted to 2D square array with paddings if required
        The output from this class can be directly used to covert to images """
    scaler = MinMaxScaler(feature_range=(0, 255))
    @staticmethod
    def required_zero(arr:np.ndarray) -> int:
        """ Function to compute how many zeros need to be appended to make no .of col square """
        assert(len(arr.shape) == 2), "Wrong shape, make sure that the input here is comming from Dataframe.to_numpy"
        length = arr.shape[1]
        sq = math.ceil(math.sqrt(length))
        return sq*sq - length
    
    @staticmethod
    def add_pad(arr: np.ndarray, pad:int) -> np.ndarray:
        """ Add 0-padding to columns axis. Adds additional pads """
        assert(len(arr.shape) == 2), "Wrong shape, make sure that the input here is comming from Dataframe.to_numpy"
        pad_col = pad
        pad_row = 0
        return np.pad(arr, [(0, pad_row), (0, pad_col)]) 

    @staticmethod
    def squareify(arr: np.ndarray) -> np.ndarray:
        """ Returns all the rows in the array reshaped as squares """
        nrows, ncols =  arr.shape
        sqrt_dim = int(math.sqrt(ncols))
        assert(sqrt_dim**2 ==  ncols), "The arrays on the row axis is not a perfect square."
        new_dim = (nrows, sqrt_dim, sqrt_dim)
        return arr.reshape(new_dim)

    @staticmethod
    def convert(df: pd.DataFrame, batch:int=3) -> np.ndarray:
        """ Main function to convert df to nd.array where each individual array is a square arrray
            ready to be converted to image """
        assert(df.shape[1] >= batch), "The now of rows of the df is less than the batch"
        arr = Df2ImageBatchNormalizer(batch).normalize(df)
        # arr = Df2ImageNormalizer().normalize(df)
        return df2array.squareify(df2array.add_pad(arr, pad=df2array.required_zero(arr)))

def _convert_to_BW(df:pd.DataFrame) -> List[Image.Image]:
    """ Converts the dataframe to a images """
    arrs = df2array.convert(df)
    return [Image.fromarray(arr, mode = 'L') for arr in arrs]

def _convert_to_RGB(df:pd.DataFrame) -> List[Image.Image]:
    arrs = df2array.convert(df)
    # To get rgb immages, we stack the 3 images in sequence on on another
    return [Image.fromarray(np.dstack((arrs[idx], arrs[idx-1], arrs[idx-2])), mode = 'RGB') for idx in range(2,arrs.shape[0])]



# Helper functions
def convert_to_images(df:pd.DataFrame, rgb=False) -> List[Image.Image]:
    # Fucntion to convert df to images
    def ensure_dtypes_ok(df: pd.DataFrame) -> bool:
        # Function to ensure that the dtypes of the cols is either float or intetger
        for dtype in df.dtypes:
            if not (np.issubdtype(dtype, np.floating) or np.issubdtype(dtype, np.integer)):
                return False    
        return True
    assert ensure_dtypes_ok(df) , "dtypes for the df are invalid" 
    return _convert_to_RGB(df) if rgb else _convert_to_BW(df)

def images_to_gif(images:List[Image.Image], gif:Path) -> None:
    # Create a gif from the images
    assert(len(images) > 1), "More images required to make gif"
    images[0].save(gif, save_all=True, append_images=[img for img in images[1:]])

def images_to_dir(images: List[Image.Image], save_dir:Path) -> None:
    # Saves all images to provided dir
    save_dir.unlink() if save_dir.exists() else save_dir.mkdir()
    for idx, image in enumerate(images):
        image.save(save_dir/f'{idx}.png')
