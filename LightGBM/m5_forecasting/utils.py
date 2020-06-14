from typing import Optional, List, Tuple

import numpy as np
import pandas as pd


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    start_mem = df.memory_usage().sum() / 1024 ** 2
    int_columns = df.select_dtypes(include=['int16', 'int32', 'int64']).columns
    float_columns = df.select_dtypes(include=["float32", 'float64']).columns

    for col in int_columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    for col in float_columns:
        df[col] = pd.to_numeric(df[col], downcast="float")

    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(
            "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df


def prepare_regression_data(data: pd.DataFrame, min_d: Optional[int] = None, max_d: Optional[int] = None,
                            useless_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    useless_cols = ["id", "date", "demand", "d", "wm_yr_wk"] \
        if useless_cols is None else useless_cols

    if min_d is not None:
        data = data[data.d >= max_d]
    if max_d is not None:
        data = data[data.d <= max_d]

    input_cols = data.columns[~data.columns.isin(useless_cols)]
    output_cols = ['demand']

    return data[input_cols], data[output_cols]


def train_val_split_random(X: pd.DataFrame, y: pd.DataFrame, num_val: int = 2_000_000):
    fake_valid_inds = np.random.choice(X.index.values, num_val, replace=False)
    train_inds = np.setdiff1d(X.index.values, fake_valid_inds)

    return X.loc[train_inds], y.loc[train_inds], X.loc[fake_valid_inds], y.loc[fake_valid_inds]


def reset_dataframe_pivot_cache(df: pd.DataFrame):
    for attr in dir(df):
        if attr.startswith('pivoted_'):
            delattr(df, attr)
