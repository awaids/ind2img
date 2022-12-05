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
    def test_normalizer(self):
        norm_df = Df2ImageNormalizer.normalize(
            df = pd.DataFrame({
            'A':[10.0, 20.0, 30.0, 50.0, 40.0],
            'B':[-1,-2,0,1,2],
            'C':[1,2,3,4,1]
            })
        )
        ref_df = pd.DataFrame({
            'A':[0, 63, 127, 255, 191],
            'B':[63, 0, 127, 191, 255],
            'C':[0, 85, 170, 255, 0]}, dtype=int)
        assert(norm_df.equals(ref_df)), "Normalized df not same as reference"

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

    def test_main_working(self):
        df = pd.DataFrame({
            'A':[10, 20, 30],
            'B':[-1, -2, -3],
            'C':[1.5, 2.5, 3.5]}).astype(float)
        arr = df2array.convert(df)
        
        assert(arr.shape == (3, 2, 2)), "Shape incorect"
        assert(np.array_equal(arr[0], np.array([[0, 255],[0, 0]]))), "Row not equal"
        assert(np.array_equal(arr[1], np.array([[127, 127],[127, 0]]))), "Row not equal"
        assert(np.array_equal(arr[2], np.array([[255, 0],[255, 0]]))), "Row not equal"

class TestDf2BW:
    def test_main_working(self):
        # Nothing should fail here
        df = pd.DataFrame({
           'A':[10.0, 20.0, 30.0, 50.0, 40.0],
           'B':[-1,-2,0,1,2],
           'C':[1,2,3,4,1]
        })
        images = Df2BW.convert(df)
        assert(len(images) == df.shape[0]), "Incorrect number of images recieved"

        for image in images:
            assert(image.size == (2, 2)), "Incorrect image size"