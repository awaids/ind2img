import pytest
import numpy as np
import pandas as pd
from array2img import *

class TestDf2ImageBatchNormalizer:
    def test_batch_nomalization(self):
        df = pd.DataFrame({
            'A':[0, 63, 127, 255, 191, 2],
            'B':[63, 0, 127, 191, 255, 2],
            'C':[0, 85, 170, 255, 0, 2]})
        batch = 3
        ref_arr = np.array(
            [
                [255, 255, 255],
                [255, 254, 255],
                [127, 255, 0],
                [0, 0, 2]
            ], 
            dtype=np.uint8
        )
        new_arr = Df2ImageBatchNormalizer(batch=batch).normalize(df)
        assert(np.array_equal(new_arr, ref_arr)), "Not matching ref"

    def test_batch_nomalization_asserts(self):
        with pytest.raises(AssertionError):
            Df2ImageBatchNormalizer(batch=1)
        
        with pytest.raises(AssertionError):
            # This assertion should fail as df not big enough
            df = pd.DataFrame({
            'A':[0, 63, 127, 255, 191, 2],
            'B':[63, 0, 127, 191, 255, 2],
            'C':[0, 85, 170, 255, 0, 2]})
            Df2ImageBatchNormalizer(batch=7).normalize(df)

        # This should pass
        df = pd.DataFrame({
            'A':[0, 63, 127],
            'B':[63, 0, 127],
            'C':[0, 85, 170]})
        arr = Df2ImageBatchNormalizer(batch=3).normalize(df)
        assert(arr.shape[0] == 1), "Incorrect number of rows found"

class Testdf2array:
    def test_required_zero(self):
        assert(df2array.required_zero(np.array([[1]])) == 0), "Required zeros must 0"
        assert(df2array.required_zero(np.array([[1,2,3], [4,5,6]])) == 1), "Required zeros must 1"

        # Check assert
        with pytest.raises(AssertionError):
            # Check wrong dimension
            df2array.required_zero(np.array([1]))
        with pytest.raises(AssertionError):
            # Check wrong dimension
            df2array.required_zero(np.array([1,2,3]))
        with pytest.raises(AssertionError):
            # Check 3D arrays
            arr = np.array([[[1],[2],[3]]])
            df2array.required_zero(arr)
        
    def test_add_pad(self):
        arr = np.array([[1]])
        ref = np.array([[1,0,0]])
        assert(np.array_equal(df2array.add_pad(arr, 2), ref)), "Invalid padding"

        arr = np.array([[1], [2]])
        ref = np.array([[1,0], [2, 0]])
        assert(np.array_equal(df2array.add_pad(arr, 1), ref)), "Invalid padding"

    def test_squareify(self):
        arr = df2array.squareify(np.array([[1]]))
        assert(np.array_equal(arr, np.array([[[1]]]))), "Concerting 2D to 3D" 

        with pytest.raises(AssertionError):
            # dimension not square must raise
            df2array.squareify(np.array([[1, 2, 3]]))
        
        # Check shape and element positions
        arr = df2array.squareify(np.array([[1, 2, 3, 4]]))
        ref = np.array([[[1, 2], [3, 4]]])
        assert(np.array_equal(arr, ref)), "Shape and elements position must be correct"   

        # Check for array with mutliple rows
        arr = df2array.squareify(np.array([[1, 2, 3, 4], ['a', 'b', 'c', 'd']]))
        ref = np.array([[[1, 2], [3, 4]], [['a', 'b'], ['c', 'd']]])
        assert(np.array_equal(arr, ref)), "Shape and elements position must be correct"   

    def test_semi_flow(self):
        # Test squareify and padding
        df = pd.DataFrame({
            'A':[10, 20, 'a'],
            'B':[-1, -2, 'b'],
            'C':[1, 2, 'c']}).astype(object)
        # Construct the reference
        arr = np.array(
            [
                [10, -1,  1],
                [20, -2,  2],
                ['a', 'b',  'c'],
            ]
        )
        pad_col, pad_row = 1, 0
        padded_arr = np.pad(arr, [(0, pad_row), (0, pad_col)])
        ref_arr = padded_arr.reshape((3, 2, 2))

        # Construct the array from df
        arr = df.to_numpy()
        new_arr = df2array.squareify(df2array.add_pad(arr, pad=df2array.required_zero(arr)))
        assert(np.array_equal(new_arr.astype(str), ref_arr.astype(str))), "Theses arrays must be the same"
        assert(len(new_arr.shape) == 3), "Shape not correct"
        assert(new_arr.shape[1] == new_arr.shape[2]), "Shape not a square"

        # check individual elements
        assert(np.array_equal(new_arr[0], np.array([[10, -1], [1, 0]])))
        assert(np.array_equal(new_arr[1], np.array([[20, -2], [2, 0]])))
        assert(np.array_equal(new_arr[2].astype(str), np.array([['a', 'b'], ['c', 0]]).astype(str)))

    def test_batch_working(self):
        df = pd.DataFrame({
            'A':[10, 20, 30],
            'B':[-1, -2, -3],
            'C':[1.5, 2.5, 3.5]}).astype(float)
        arr = df2array.convert(df)
        
        assert(arr.shape == (1, 2, 2)), "Shape incorect"
        assert(np.array_equal(arr[0], np.array([[255, 0],[255, 0]]))), "Row not equal"
    
    def test_dtype_asserts(self):
        # Test if there is a dtype that is not supported in the df asserted out
        df = pd.read_csv(Path(__file__).parent / "short_df.csv")
        with pytest.raises(AssertionError):
            convert_to_images(df, rgb=False)
        # This should pass
        convert_to_images(df.drop(columns=['Date']), rgb=False)


def test_Df2BW_working():
    # Test the main working of the module
    df = pd.DataFrame({
        'A':[10.0, 20.0, 30.0, 50.0, 40.0],
        'B':[-1,-2,0,1,2],
        'C':[1,2,3,4,1]
    })
    images = convert_to_images(df, rgb=False)
    # -2 here as the Df2ImageBatchNormalizer has a batch 3
    assert(len(images) == df.shape[0] - 2), "Incorrect number of images recieved"

    # Check the sizes of all images
    for image in images:
        assert(image.size == (2, 2)), "Incorrect image size"

def test_convert_images():
    ref_dim = (3, 3)
    # Read the input
    _ref_dir = Path(__file__).parent / 'ref'
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
    # Test if the images are properly covnerted to gif
    _ref_dir = Path(__file__).parent / 'ref'
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
    _ref_dir = Path(__file__).parent / 'ref'
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
