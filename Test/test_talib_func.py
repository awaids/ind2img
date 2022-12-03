import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from talib import abstract, MACD
from talib_indicators import Talib_func, CumputeTALibIndicators, get_valid_talib_functions

_BTC_CSV = Path(__file__).parent / 'BTC-USD_1d_yahoo.csv'

def read_csv(csv:Path) -> pd.DataFrame:
    return pd.read_csv(csv)

class TestTalibfunc:
    def test_basic_working(self):
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
        args = {'timeperiod' : 5}
        result = rsi(df=df, **args)
        val = result[rsi.output_names[0]]
        assert(np.allclose(ref, val, equal_nan=True)), "Both values must be the same"

    def test_out_names(self):
        # Test the required names for the talib function

        assert(Talib_func('bbands').output_names == ['BBANDS_upperband', 'BBANDS_middleband', 'BBANDS_lowerband']), "Wrong output names"
        assert(Talib_func('SAR').output_names == ['SAR_real']), "Wrong output names"

    def test_multiple_outputs(self):
        # Need to capitalize the column names as this is what we normally get
        df = read_csv(_BTC_CSV)
        ref = MACD(df['Close'].to_numpy())
        macd = Talib_func('macd')
        result = macd(df=df)
        
        for r, v in zip(ref, result.values()):
            assert(np.allclose(r, v, equal_nan=True)), "Multiple output ref mis-match"

    def test_parameters(self):
        macd = Talib_func('macd')
        assert(set(macd.paramters_names) == {'fastperiod', 'slowperiod', 'signalperiod'}), 'keys dont match for the parameters'

        add = Talib_func('add')
        assert(len(add.paramters_names) == 0), "This should be empty"

    def test_has_timeperiod(self):
        """ Check if an indicator is configureable with time period """
        assert(Talib_func('RSI').has_timeperiod == True)
        assert(Talib_func('ADOSC').has_timeperiod == False)


def test_get_valid_talib_functions():
    """ Test is the functions found are the one we are currently adding """
    valid_functions = set(get_valid_talib_functions())
    assert(valid_functions == set([
        'HT_DCPERIOD', 'HT_DCPHASE', 'HT_PHASOR', 'HT_SINE', 'HT_TRENDMODE', 'ADD', 'DIV',
        'MAX', 'MAXINDEX', 'MIN', 'MININDEX', 'MINMAX', 'MINMAXINDEX', 'MULT', 'SUB', 'SUM',
        'ATAN', 'CEIL', 'COS', 'COSH', 'EXP', 'FLOOR', 'LN', 'LOG10', 'SIN', 'SINH', 'SQRT',
        'TAN', 'TANH', 'ADX', 'ADXR', 'APO', 'AROON', 'AROONOSC', 'BOP', 'CCI', 'CMO', 'DX',
        'MACD', 'MACDEXT', 'MACDFIX', 'MFI', 'MINUS_DI', 'MINUS_DM', 'MOM', 'PLUS_DI', 'PLUS_DM',
        'PPO', 'ROC', 'ROCP', 'ROCR', 'ROCR100', 'RSI', 'STOCH', 'STOCHF', 'STOCHRSI', 'TRIX',
        'ULTOSC', 'WILLR', 'BBANDS', 'DEMA', 'EMA', 'KAMA', 'MA', 'MAMA', 'MIDPOINT', 'MIDPRICE',
        'SAR', 'SAREXT', 'SMA', 'TEMA', 'TRIMA', 'WMA', 'CDL2CROWS', 'CDL3BLACKCROWS', 'CDL3INSIDE',
        'CDL3LINESTRIKE', 'CDL3OUTSIDE', 'CDL3STARSINSOUTH', 'CDL3WHITESOLDIERS', 'CDLABANDONEDBABY',
        'CDLADVANCEBLOCK', 'CDLBELTHOLD', 'CDLBREAKAWAY', 'CDLCLOSINGMARUBOZU', 'CDLCONCEALBABYSWALL',
        'CDLCOUNTERATTACK', 'CDLDARKCLOUDCOVER', 'CDLDOJI', 'CDLDOJISTAR', 'CDLDRAGONFLYDOJI',
        'CDLENGULFING', 'CDLEVENINGDOJISTAR', 'CDLEVENINGSTAR', 'CDLGAPSIDESIDEWHITE', 'CDLGRAVESTONEDOJI',
        'CDLHAMMER', 'CDLHANGINGMAN', 'CDLHARAMI', 'CDLHARAMICROSS', 'CDLHIGHWAVE', 'CDLHIKKAKE',
        'CDLHIKKAKEMOD', 'CDLHOMINGPIGEON', 'CDLIDENTICAL3CROWS', 'CDLINNECK', 'CDLINVERTEDHAMMER',
        'CDLKICKING', 'CDLKICKINGBYLENGTH', 'CDLLADDERBOTTOM', 'CDLLONGLEGGEDDOJI', 'CDLLONGLINE',
        'CDLMARUBOZU', 'CDLMATCHINGLOW', 'CDLMATHOLD', 'CDLMORNINGDOJISTAR', 'CDLMORNINGSTAR', 'CDLONNECK',
        'CDLPIERCING', 'CDLRICKSHAWMAN', 'CDLRISEFALL3METHODS', 'CDLSEPARATINGLINES', 'CDLSHOOTINGSTAR',
        'CDLSHORTLINE', 'CDLSPINNINGTOP', 'CDLSTALLEDPATTERN', 'CDLSTICKSANDWICH', 'CDLTAKURI', 'CDLTASUKIGAP',
        'CDLTHRUSTING', 'CDLTRISTAR', 'CDLUNIQUE3RIVER', 'CDLUPSIDEGAP2CROWS', 'CDLXSIDEGAP3METHODS',
        'AVGPRICE', 'MEDPRICE', 'TYPPRICE', 'WCLPRICE', 'BETA', 'CORREL', 'LINEARREG', 'LINEARREG_ANGLE',
        'LINEARREG_INTERCEPT', 'LINEARREG_SLOPE', 'STDDEV', 'TSF', 'VAR', 'ATR', 'NATR', 'TRANGE', 'AD',
        'ADOSC', 'OBV'])), "The functions found are not the same"

class TestCumputeTALibIndicators:
    def test_valid_talib_functions_min_period(self):
        """ This test tries to manually caluclaute the required minimum period for the functions returned by test_get_valid_talib_functions"""
        df = pd.read_csv(Path(__file__).parent / 'BTC-USD_1d_yahoo.csv')
        assert(df.shape[0] > 500), "No. of rows required for the test must be greater than 500"

        # Helper function
        def get_func_size(df:pd.DataFrame, timeperiod:int) -> Dict[str, int]:
            fmap = {}
            # for func in ['ADXR', 'TRIX']:
            for func in get_valid_talib_functions():
                outputs = Talib_func(func)(df=df, timeperiod=timeperiod)
                min_rows = min([len(val[~np.isnan(val)]) for val in outputs.values()])
                fmap[func] = min_rows
            return fmap

        # Helper function
        def find_for_timeperiod(timeperiod:int) -> Tuple[int, List[str]]:
            """ returns the min required rows for a give period """
            rows = timeperiod
            for _ in range(df.shape[0]):
                df_sliced = df.iloc[:rows]
                fmap = get_func_size(df=df_sliced, timeperiod=timeperiod)
                if min(fmap.values()) == 0:
                    rows = rows + 1
                else:
                    return rows, [k for k, v in fmap.items() if v == 1]
            assert(False), "Timeperiod not found!"
        
        with pytest.raises(AssertionError):
            # This should cause an init() asssertion
            CumputeTALibIndicators([1])

        for tp in [2, 10, 50, 100, 150, 500]:
            rtp, _ = find_for_timeperiod(tp)
            assert(CumputeTALibIndicators([tp]).minimum_period_required == rtp), "Minimum timeperiod calculation is incorrect, Make sure that get_valid_talib_functions() is still unchanged"

    def test_get_columns_list(self):
        """ Check the working of get_columns_list """
        assert(len(CumputeTALibIndicators(timeperiods=[10]).get_columns_list()) == 170), "Number of columns to be added not the same"
        assert(len(CumputeTALibIndicators(timeperiods=[10, 20]).get_columns_list()) == 170 + 56), "Number of columns to be added not the same"
        assert(len(CumputeTALibIndicators(timeperiods=[10, 20, 30]).get_columns_list()) == 170 + 56*2), "Number of columns to be added not the same"

    def test_original_df_notchanged(self):
        """ Test if the orignal does not change """
        original = pd.read_csv(Path(__file__).parent / 'BTC-USD_1d_yahoo.csv')
        df = original.copy(deep=True)
        indA =  CumputeTALibIndicators(timeperiods=[10])
        indA.get_indicators_df(df)
        assert(original.equals(df)), "Original df has been changed"
    
    def test_CumputeTALibIndicators(self):
        """ Test main working """
        df = pd.read_csv(Path(__file__).parent / 'BTC-USD_1d_yahoo.csv')
        indA =  CumputeTALibIndicators(timeperiods=[10])
        ind_df = indA.get_indicators_df(df.iloc[: indA.minimum_period_required])

        # Check if only no. of rows expected is correct
        assert(ind_df.shape[0] == 1), "Dataframe must contain exaclty one row here"
        # Assert all required ind columns are there
        assert(set(ind_df.columns).difference(set(indA.get_columns_list())) == {'High', 'Volume', 'Low', 'Close', 'Open'}), "Required indicator columns are not properly added"
        # Check if the number of row 
        assert(len(ind_df.columns) == 175), "Unexpected number of columns"
    
    def test_all_data_present(self):
        """ Make sure the right no. of rows are present after computing indicators """
        df = pd.read_csv(Path(__file__).parent / 'BTC-USD_1d_yahoo.csv')
        indA =  CumputeTALibIndicators(timeperiods=[10, 100])
        ind_df = indA.get_indicators_df(df)

        assert(ind_df.shape[0] == df.shape[0] - indA.minimum_period_required + 1), "There are rows missing!"
    




