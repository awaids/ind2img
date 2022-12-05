import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from array2img import img2gif, img2dir
from array2img import Df2BW, Df2RGB
from talib_indicators import CumputeTALibIndicators


def test_df2BWimages():
    ref_dim = (17, 17)
    ref_imgs = 2653

    curr_dir = Path(__file__).parent
    df = pd.read_csv(curr_dir / 'BTC-USD_1d_yahoo.csv')
    ind_df = CumputeTALibIndicators(timeperiods=[10,50,100]).get_indicators_df(df)

    # Checking the bw images
    images_bw = Df2BW.convert(ind_df)
    assert(images_bw[0].size == ref_dim), "Image dimensions not correct for the given timeperiods"
    assert(len(images_bw) == ref_imgs), "No. of images are incorrect"
    # images_bw[0].save(curr_dir / 'test_bw.png') # Update the reference
    arr = np.asarray(images_bw[0])
    ref_arr = np.asarray(Image.open(curr_dir / 'test_bw.png'))
    assert(np.array_equal(arr, ref_arr)), "The arrays generated are not same as the reference"

    # Checking the bw images
    images_rgb = Df2RGB.convert(ind_df)
    assert(images_rgb[0].size == ref_dim), "Image dimensions not correct for the given timeperiods"
    assert(len(images_rgb) == ref_imgs - 2), "No. of images are incorrect"
    # images_rgb[0].save(curr_dir / 'test_rgb.png') # Update the reference
    arr = np.asarray(images_rgb[0])
    ref_arr = np.asarray(Image.open(curr_dir / 'test_rgb.png'))
    assert(np.array_equal(arr, ref_arr)), "The arrays generated are not same as the reference"
    
    
def test_main_working():
    curr_dir = Path(__file__).parent
    df = pd.read_csv(curr_dir / 'BTC-USD_1d_yahoo.csv')
    ind_df = CumputeTALibIndicators(timeperiods=list(range(2,300))).get_indicators_df(df)

    img2gif(Df2RGB.convert(ind_df), curr_dir / 'rgb.gif')
    img2gif(Df2BW.convert(ind_df), curr_dir / 'bw.gif')
    print("Done!")