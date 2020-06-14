import pandas as pd
import numpy as np
import os
from collections import namedtuple

from typing import Optional

from m5_forecasting.utils import reduce_mem_usage

DataTuple = namedtuple('DataTuple', ['calendar', 'prices', 'sales'])


def load_data_raw(base_path: Optional[str] = None, read_csvs: bool = False, evaluation: bool = False) -> DataTuple:
    base_path = "data" if base_path is None else base_path
    if read_csvs:
        calendar_csv = os.path.join(base_path, "calendar.csv")
        prices_csv = os.path.join(base_path, "sell_prices.csv")
        if evaluation:
            sales_csv = os.path.join(base_path, "sales_train_evaluation.csv")
        else:
            sales_csv = os.path.join(base_path, "sales_train_validation.csv")

        calendar = pd.read_csv(calendar_csv).pipe(reduce_mem_usage)
        prices = pd.read_csv(prices_csv).pipe(reduce_mem_usage)
        sales = pd.read_csv(sales_csv).pipe(reduce_mem_usage)
    else:
        data_path = os.path.join(base_path, "data.h5")

        calendar = pd.read_hdf(data_path, 'calendar')
        assert isinstance(calendar, pd.DataFrame)
        calendar = reduce_mem_usage(calendar)

        prices = pd.read_hdf(data_path, 'prices')
        assert isinstance(prices, pd.DataFrame)
        prices = reduce_mem_usage(prices)

        sales = pd.read_hdf(data_path, 'sales')
        assert isinstance(sales, pd.DataFrame)
        sales = reduce_mem_usage(sales)

    return DataTuple(calendar, prices, sales)


def load_data():
    calendar, prices, sales = load_data_raw(read_csvs=True)

    calendar.date = pd.to_datetime(calendar.date)
    calendar.d = calendar.d.str.extract(r"(\d+)").astype(np.int16)
    calendar.drop(["weekday", "wday", "month", "year"], axis=1, inplace=True)

    calendar = reduce_mem_usage(calendar)
    prices = reduce_mem_usage(prices)
    sales = reduce_mem_usage(sales)

    return DataTuple(calendar, prices, sales)


def load_data_merged() -> pd.DataFrame:
    data = pd.read_hdf('data/data.h5', key='data')
    assert isinstance(data, pd.DataFrame)
    return data
