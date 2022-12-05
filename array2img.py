import math
import numpy as np
import pandas as pd
from PIL import Image

from pathlib import Path
from functools import cache
from typing import List
from sklearn.preprocessing import MinMaxScaler

def img2gif(images:List[Image.Image], gif:Path) -> None:
    # Create a gif from the images
    assert(len(images) > 1), "More images required to make gif"
    images[0].save(gif, save_all=True, append_images=[img for img in images[1:]])

def img2dir(images: List[Image.Image], save_dir:Path) -> None:
    # Saves all images to provided dir
    save_dir.unlink() if save_dir.exists() else save_dir.mkdir()
    for idx, image in enumerate(images):
        image.save(save_dir/f'{idx}.png')

class Df2ImageNormalizer:
    scaler = MinMaxScaler(feature_range=(0, 255))
    @staticmethod
    def normalize(df:pd.DataFrame) -> pd.DataFrame:
        """ Normalize the df column wise to for images """
        return pd.DataFrame(Df2ImageNormalizer.scaler.fit_transform(df[df.columns]), columns=df.columns).astype(int)

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
    def normalize(df:pd.DataFrame) -> pd.DataFrame:
        """ Normalize the df column wise to for images """
        return pd.DataFrame(Df2ImageNormalizer.scaler.fit_transform(df[df.columns].astype(float)), columns=df.columns)

    @staticmethod
    def convert(df: pd.DataFrame) -> np.ndarray:
        """ Main function to convert df to nd.array where each individual array is a square arrray
            ready to be converted to image """
        arr = df2array.normalize(df).to_numpy().astype(np.uint8)
        return df2array.squareify(df2array.add_pad(arr, pad=df2array.required_zero(arr)))

class Df2BW:
    """ Class to convert df to BW images """
    @staticmethod
    def convert(df:pd.DataFrame) -> List[Image.Image]:
        arrs = df2array.convert(df)
        return [Image.fromarray(arr, mode = 'L') for arr in arrs]

class Df2RGB:
    """ Class to convert df to BW images """
    @staticmethod
    def convert(df:pd.DataFrame) -> List[Image.Image]:
        arrs = df2array.convert(df)
        return [Image.fromarray(np.dstack((arrs[idx], arrs[idx-1], arrs[idx-2])), mode = 'RGB') for idx in range(2,arrs.shape[0])]