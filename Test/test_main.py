import pytest
import pandas as pd
from pathlib import Path
from talib_indicators import CumputeTALibIndicators
from array2img import convert_to_images
from array2img import images_to_gif
from array2img import images_to_dir

class Test_main_working:
    def test_steps(self):
        curr_dir = Path(__file__).parent
        df = pd.read_csv(curr_dir / 'BTC-USD_1d_yahoo.csv')

        # Setup the indicators and its timeperiods
        timeperiods = list(range(2, 500))
        cti = CumputeTALibIndicators(timeperiods=timeperiods)

        # Get df with indicator data
        ind_df = cti.get_indicators_df(df=df)
        assert(df.shape[0] - cti.minimum_period_required + 1 == ind_df.shape[0]), "Unexpected df rows"

        # Test bw images generation
        images_bw = convert_to_images(df=ind_df, rgb=False)
        assert(len(images_bw) == ind_df.shape[0] - 2), "Unexpected number of images"

        # Test rgb images generation
        images_rgb = convert_to_images(df=ind_df, rgb=True)
        assert(len(images_rgb) == ind_df.shape[0] - 4), "Unexpected number of images"

        # create gifs
        images_to_gif(images=images_bw, gif=curr_dir/"bw.gif")
        images_to_gif(images=images_rgb, gif=curr_dir/"rgb.gif")

    