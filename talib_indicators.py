import numpy as np
import pandas as pd
from talib import abstract, get_function_groups, __version__
from functools import cache
from collections import OrderedDict
from typing import List, Dict

@cache
def get_valid_talib_functions() -> List[str]:
    """ Helper function to return all the valid functions from talib """
    assert(__version__ == '0.4.24'), "Talib version not the same!"
    # TODO: Make this into a configuration
    VALID_GROUPS = [
        'Overlap Studies', 'Momentum Indicators', 'Pattern Recognition', 
        'Price Transform', 'Volatility Indicators', 'Volume Indicators',
        'Cycle Indicators', 'Math Operators', 'Math Transform', 'Statistic Functions'
    ]
    IGNORE_INDICATORS = ['MAVP', 'T3', 'HT_TRENDLINE', 'ACOS', 'ASIN', 'COSH', 'EXP', 'SINH']
    valid_funcs = []
    for group, funcs in get_function_groups().items():
        if group not in VALID_GROUPS:
            continue
        for func in funcs:
            if func in IGNORE_INDICATORS:
                continue
            valid_funcs.append(func)
    return valid_funcs

class Talib_func:
    """ Class to maintian the abstract api for talib  """
    def __init__(self, ind_name:str) -> None:
        self.func = abstract.Function(ind_name)
        self.info = self.func.info
    
    def __call__(self, df:pd.DataFrame, **kwargs) -> Dict[str, np.ndarray]:
        # Make the function callable
        cols = ['Open', 'Close', 'High', 'Low', 'Volume']
        assert(all([True if col in df.columns else False for col in cols])), "Missing columns in df"
        inputs= {col.lower() : df[col].to_numpy(dtype=float) for col in cols}
        output = self.func(inputs, **kwargs)
        if len(self.output_names) > 1:
            # Use zip only when we have more than 1 output columns
            return {col : val for col, val in zip(self.output_names, output)}
        return {self.output_names[0] : output}

    @property
    def name(self) -> str:
        return self.info.get('name')

    @property
    def inputs(self) -> OrderedDict:
        return self.info.get('input_names')

    @property
    def output_names(self) -> List[str]:
        return [f'{self.name}_{col}' for col in self.info.get('output_names')]

    @property
    def group(self) -> str:
        return self.info.get('group')

    @property
    def parameters(self) -> OrderedDict:
        return OrderedDict(self.info.get('parameters'))

    @property
    def has_timeperiod(self) -> bool:
        """ if true than it mean that we this indicator can be confiured by timeperiod """
        return True if 'timeperiod' in self.paramters_names else False

    @property
    def paramters_names(self) -> List[str]:
        # Returns a list of paramters names
        return list(self.parameters.keys())

class CumputeTALibIndicators:
    """ Use this class to compute all indicator known to TALib """
    def __init__(self, timeperiods:List[int]) -> None:
        assert(min(timeperiods) >= 2), "Minimum timeperiod valid is 2"
        self.timeperiods = timeperiods
        self.minimum_period_required = max(64, 3 * max(timeperiods) - 1)

    def _get_ouput_dict(self, output:dict, suffix:str='') -> Dict[str, np.ndarray]:
        """ Iterates over the result from talib function and returns them as a dict """
        outputs = {}
        for col, val in output.items():
            outputs[f'{col}{suffix}'] = val
        return outputs

    def _get_timeperiod_dict(self, df:pd.DataFrame, func:Talib_func) -> Dict[str, np.ndarray]:
        """ Returns adf with all the indicator value with all different timeperiods """
        outputs = {}
        for tp in self.timeperiods:
            output = func(df=df, timeperiod=tp)
            outputs = {**outputs, **self._get_ouput_dict(output, tp)}
        return outputs

    def _get_ind_dict(self, func:str, df:pd.DataFrame) -> Dict[str, np.ndarray]:
        """ Appends the indicator data  """
        tafunc = Talib_func(ind_name=func)
        # Check if the indicator has timeperiods input that can be tweaked
        if tafunc.has_timeperiod:
            outputs = self._get_timeperiod_dict(df, tafunc)
        else:
            outputs = tafunc(df=df)
        return outputs

    def get_columns_list(self) -> List[str]:
        """ Returns a list of all the columns that will be calculated """
        output_cols = []
        for func in get_valid_talib_functions():
            tafunc = Talib_func(func)
            if tafunc.has_timeperiod:
                for tp in self.timeperiods:
                    for col in tafunc.output_names:
                        output_cols.append(f'{col}{tp}')
            else:
                output_cols = output_cols + tafunc.output_names
        return sorted(output_cols)
        
    # Appends indicator data to current df
    def get_indicators_df(self, df: pd.DataFrame, dropNaNs = True) -> pd.DataFrame:
        """ Returns a new df with the indicator data concatenated with original df.
            The returned df has columns sorted """
        cols = {'Open', 'Close', 'High', 'Low', 'Volume'}
        # Check for sanity of the df
        assert(cols.issubset(set(df.columns))), "Required columns missing in df"
        assert(df.shape[0] >= self.minimum_period_required), "Minimum required period is not satisfied"
        ind_dict = {}
        for func in get_valid_talib_functions():
            ind_dict = {**ind_dict, **self._get_ind_dict(func=func, df=df)}
        drop_columns= set(df.columns.difference(cols))
        ind_df = pd.concat([df.copy(deep=True).drop(columns=drop_columns), pd.DataFrame.from_dict(ind_dict)], axis=1)
        if dropNaNs:
            ind_df.dropna(inplace=True)
        return ind_df[sorted(ind_df.columns)]