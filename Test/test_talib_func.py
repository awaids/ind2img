import numpy as np
import pandas as pd
from pathlib import Path
from talib import abstract, MACD
from talib_indicators import Talib_func

_BTC_CSV = Path(__file__).parent / 'BTC-USD_1d_yahoo.csv'

def read_csv(csv:Path) -> pd.DataFrame:
    return pd.read_csv(csv)

def test_basic_working():
    # Test if Talib_func is callable with Dataframes
    inputs = {
        'open': np.random.random(10),
        'high': np.random.random(10),
        'low': np.random.random(10),
        'close': np.random.random(10),
        'volume': np.random.random(10)
    }
    # Need to capitalize the column names as this is what we normally get
    df = pd.DataFrame.from_dict(inputs).rename(lambda x:x.capitalize(), axis='columns')
    ref = abstract.Function('rsi')(inputs, timeperiod = 5)
    rsi = Talib_func('rsi')
    result = rsi(df=df, timeperiod = 5)
    val = result[rsi.output_names[0]]
    assert(np.allclose(ref, val, equal_nan=True)), "Both values must be the same"

def test_out_names():
    # Test the required names for the talib function

    assert(Talib_func('bbands').output_names == ['BBANDS_upperband', 'BBANDS_middleband', 'BBANDS_lowerband']), "Wrong output names"
    assert(Talib_func('SAR').output_names == ['SAR_real']), "Wrong output names"

def test_multiple_outputs():
    # Need to capitalize the column names as this is what we normally get
    df = read_csv(_BTC_CSV)
    ref = MACD(df['Close'].to_numpy())
    macd = Talib_func('macd')
    result = macd(df=df)
    
    for r, v in zip(ref, result.values()):
        assert(np.allclose(r, v, equal_nan=True)), "Multiple output ref mis-match"
