import numpy as np
import pandas as pd
from talib import abstract
from typing import List, Dict

class Talib_func:
    """ Class to maintian the abstract api for talib  """
    def __init__(self, ind_name:str) -> None:
        self.func = abstract.Function(ind_name)
        self.info = self.func.info
    
    def __call__(self, df:pd.DataFrame, **kwargs) -> Dict[str, np.ndarray]:
        # Make the function callable
        cols = ['Open', 'Close', 'High', 'Low', 'Volume']
        assert(all([True if col in df.columns else False for col in cols])), "Missing columns in df"
        inputs= {col.lower() : df[col].to_numpy() for col in cols}
        output = self.func(inputs, **kwargs)
        if len(self.output_names) > 1:
            # Use zip only when we have more than 1 output columns
            return {col : val for col, val in zip(self.output_names, output)}
        return {self.output_names[0] : output}

    @property
    def name(self) -> str:
        return self.info.get('name')

    @property
    def output_names(self) -> List[str]:
        return [f'{self.name}_{col}' for col in self.info.get('output_names')]

    @property
    def group(self) -> str:
        return self.info.get('group')