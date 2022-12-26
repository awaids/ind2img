import pandas as pd
import numpy as np
from PIL import Image
from array2img import convert_to_images
from array2img import images_to_gif
from array2img import images_to_dir
from pathlib import Path

_ref_dir = Path(__file__).parent / 'ref'

def test_convert_images():
    # Read the input
    ref_dim = (3, 3)
    df = pd.read_csv(_ref_dir / 'ref_image_df.csv')
    df_rows = df.shape[0]

    # Checking the bw images
    images = convert_to_images(df, rgb=False)
    assert len(images) ==  df_rows - 2 , "Images sizes incorrect"
    assert images[0].size == ref_dim , "Image dimensions not correct"
    
    # Enable to update reference
    # images[0].save(_ref_dir / f'bw_ref_0.png')
    im = Image.open(_ref_dir / f'bw_ref_0.png') 
    assert np.array_equal(np.asarray(im), np.asarray(images[0])), "Reference not the same"

    # Checking the rgb images
    images = convert_to_images(df, rgb=True)
    assert len(images) ==  df_rows - 4 , "Images sizes incorrect"
    assert images[0].size == ref_dim , "Image dimensions not correct"
    
    # Enable to update reference
    # images[0].save(_ref_dir / f'rgb_ref_0.png')
    im = Image.open(_ref_dir / f'rgb_ref_0.png') 
    assert np.array_equal(np.asarray(im), np.asarray(images[0])), "Reference not the same"

def test_images_to_gif():
    # # Test if the images are properly covnerted to gif
    images = [Image.open(_ref_dir / path) for path in ['bw_ref_0.png', 'rgb_ref_0.png']]
    gif_path = Path(_ref_dir / 'tmp.gif')
    images_to_gif(images=images, gif=gif_path)
    assert gif_path.exists() , "Gif not created"
    
    # Test the gif
    gif = Image.open(gif_path)
    assert gif.n_frames == 2, "Gif does not have correct number of frames"

    # Cleanup
    gif.close()
    gif_path.unlink()

def test_images_to_dir():
    # Test if the image dump is proper
    save_dir = Path(__file__).parent / 'tmp_dir'
    images = [Image.open(_ref_dir / path) for path in ['bw_ref_0.png', 'rgb_ref_0.png']]
    images_to_dir(images, save_dir)
    assert save_dir.exists(), "save_dir not created"
    assert len(list(save_dir.iterdir())) == len(images), "Expected number of images not found"

    # clean up
    for file in save_dir.iterdir():
        file.unlink()
    save_dir.rmdir()
