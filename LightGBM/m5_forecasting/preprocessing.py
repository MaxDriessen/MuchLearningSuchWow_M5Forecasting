from typing import Union, List, Optional

import dask.dataframe as dd
import numpy as np
import pandas as pd
from numba import njit, prange

from m5_forecasting.loader import DataTuple, reduce_mem_usage

DAYS_TO_PREDICT = 28

DataType = Union[DataTuple, pd.DataFrame]


class PreprocessingStep(object):
    def __init__(self):
        pass

    def __call__(self, data: DataType) -> DataType:
        return self.process(data)

    def process(self, data: DataType) -> DataType:
        if isinstance(data, pd.DataFrame):
            return self.process_merged(data)
        elif isinstance(data, tuple) and list(map(type, data)) == [pd.DataFrame] * 3:
            return self.process_separate(data)
        else:
            raise ValueError(f'Invalid type for data: {type(data)}')

    def process_merged(self, data: pd.DataFrame) -> DataType:
        raise NotImplementedError()

    def process_separate(self, data: DataTuple) -> DataType:
        raise NotImplementedError()

    def _raise_invalid_type(self, x):
        raise ValueError(f'Invalid type for data: {type(x)}')


class TemporallyCropData(PreprocessingStep):
    def __init__(self, t_min: Optional[int] = None, t_max: Optional[int] = None):
        super().__init__()
        self.t_min = t_min
        self.t_max = t_max

    def process_merged(self, data: pd.DataFrame) -> DataType:
        query = np.zeros((len(data),), dtype=bool)
        if self.t_min is not None:
            query |= data.d < self.t_min
        if self.t_max is not None:
            query |= data.d > self.t_max

        data.drop(data[query].index, axis=0, inplace=True)

        return data

    def process_separate(self, data: DataTuple) -> DataType:
        calendar, prices, sales = data
        current_days = pd.to_numeric(sales.columns.str.extract(r'(\d+)').values[6:].ravel(), downcast='integer')
        days_to_remove = np.setdiff1d(current_days,
                                      np.arange(0 if self.t_min is None else self.t_min,
                                                (current_days.max() if self.t_max is None else self.t_max) + 1))
        sales.drop(columns=[f'd_{d}' for d in days_to_remove], inplace=True)
        return data


class DropSeries(PreprocessingStep):
    def __init__(self, subsample_size: int):
        super().__init__()
        self.subsample_size = subsample_size

    def process_merged(self, data: pd.DataFrame) -> DataType:
        return self._raise_invalid_type(data)

    def process_separate(self, data: DataTuple) -> DataType:
        calendar, prices, sales = data
        indices = sales.index.values
        to_keep = np.random.choice(indices, size=self.subsample_size, replace=False)
        return DataTuple(calendar, prices, sales.loc[to_keep])


class ExtendData(PreprocessingStep):
    def __init__(self, n: Optional[int] = None, upto: Optional[int] = None):
        super().__init__()
        assert n is not None or upto is not None
        self.n = n
        self.upto = upto

    def process_merged(self, data: pd.DataFrame) -> DataType:
        upto = self.upto if self.upto is not None else data.d.max() + self.n
        for i in range(data.d.max()+1, upto+1):
            new = data[data.d == data.d.max()].copy()
            new['d'] = i
            data = pd.concat([data, new], axis=0, sort=False)

        return data

    def process_separate(self, data: DataTuple) -> DataType:
        max_day = int(data.sales.columns[-1][2:])
        if self.n is not None:
            for d in range(max_day + 1, max_day + self.n + 1):
                data.sales[f'd_{d}'] = np.nan
        else:
            for d in range(max_day + 1, self.upto + 1):
                data.sales[f'd_{d}'] = np.nan

        return data


class ToDask(PreprocessingStep):
    def __init__(self):
        super().__init__()

    def process_merged(self, data: pd.DataFrame) -> DataType:
        return dd.from_pandas(data, npartitions=6)

    def process_separate(self, data: DataTuple) -> DataType:
        return self._raise_invalid_type(data)


class DownCastTypes(PreprocessingStep):
    def __init__(self):
        super().__init__()

    def process_merged(self, data: pd.DataFrame) -> DataType:
        return reduce_mem_usage(data)

    def process_separate(self, data: DataTuple) -> DataType:
        return DataTuple(reduce_mem_usage(data.calendar), reduce_mem_usage(data.prices),
                         reduce_mem_usage(data.sales))


class ToLongFormat(PreprocessingStep):
    def __init__(self):
        super().__init__()

    def process_merged(self, data: pd.DataFrame) -> DataType:
        return self._raise_invalid_type(data)

    def process_separate(self, data: DataTuple) -> DataType:
        sales = data.sales
        id_columns = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
        sales = sales.melt(id_vars=id_columns, var_name="d", value_name="demand").pipe(reduce_mem_usage)
        return DataTuple(data.calendar, data.prices, sales)


class MergeData(PreprocessingStep):
    def __init__(self):
        super().__init__()

    def process_merged(self, data: pd.DataFrame) -> DataType:
        return self._raise_invalid_type(data)

    def process_separate(self, data: DataTuple) -> DataType:
        calendar, prices, sales = data

        sales.d = sales.d.str.extract(r"(\d+)").astype(np.int16)

        calendar.date = pd.to_datetime(calendar.date)
        calendar.d = calendar.d.str.extract(r"(\d+)").astype(np.int16)
        calendar.drop(["weekday", "wday", "month", "year"], axis=1, inplace=True)

        data = sales.merge(calendar, how='left', on='d', copy=False)
        data = data.merge(prices, how='left', on=["store_id", "item_id", "wm_yr_wk"], copy=False)
        return data


class AddShiftFeatures(PreprocessingStep):
    def __init__(self, shifts: List[int], column: str = "demand"):
        super().__init__()
        self.shifts = shifts
        self.column = column

    def process_merged(self, data: pd.DataFrame) -> DataType:
        for shift in self.shifts:
            data[f"shift_{self.column}_t{shift}"] = data.groupby(["id"])[self.column].shift(shift)

        return data

    def process_separate(self, data: DataTuple) -> DataType:
        return self._raise_invalid_type(data)


@njit
def efficient_rolling_mean(x, shift, window_size):
    result = np.empty_like(x)
    result.fill(np.nan)
    for i in range(window_size - 1 + shift, len(x)):
        result[i] = x[i - shift - window_size + 1:i - shift + 1].mean()

    return result


@njit(parallel=True)
def mean_windowing(data, shift, window_size):
    result = np.empty_like(data)
    result.fill(np.nan)
    for i in prange(window_size - 1 + shift, data.shape[1]):
        for j in prange(data.shape[0]):
            result[j, i] = data[j, i - shift - window_size + 1:i - shift + 1].mean()

    return result


@njit(parallel=True)
def std_windowing(data, shift, window_size):
    result = np.empty_like(data)
    result.fill(np.nan)
    for i in prange(window_size - 1 + shift, data.shape[1]):
        for j in prange(data.shape[0]):
            mean = data[j, i - shift - window_size + 1:i - shift + 1].mean()
            result[j, i] = 0
            for k in range(window_size):
                result[j, i] += (mean - data[j, i - shift - window_size + 1 + k])**2
            result[j, i] /= (window_size-1)
            result[j, i] = np.sqrt(result[j, i])

    return result


class AddRollFeatures(PreprocessingStep):
    def __init__(self, window_sizes: List[int], type: str, column: str = "demand", shift: int = DAYS_TO_PREDICT,
                 fast: bool = True):
        super().__init__()
        self.window_sizes = window_sizes
        self.type = type
        self.column = column
        self.shift = shift
        self.fast = fast

    def process_merged(self, data: pd.DataFrame) -> DataType:
        for window in self.window_sizes:
            rolling_col_name = self._rolling_col_name(window)

            if self.fast and self.type == 'mean':
                pivoted = self._get_pivoted(data)
                data[rolling_col_name] = mean_windowing(pivoted.astype(np.float32), self.shift, window).ravel('F')
            elif self.fast and self.type == 'std':
                pivoted = self._get_pivoted(data)
                data[rolling_col_name] = std_windowing(pivoted.astype(np.float32), self.shift, window).ravel('F')
            else:
                data[rolling_col_name] = data.groupby(["id"])[
                    self.column].transform(
                    lambda x: getattr(x.shift(self.shift).rolling(window), self.type)()
                ).astype(np.float32)
        return data

    def _rolling_col_name(self, window: int):
        return f"rolling_{self.column}_{self.type}_t{window}_{self.shift}"

    def process_separate(self, data: DataTuple) -> DataType:
        return self._raise_invalid_type(data)

    def _get_pivoted(self, data: pd.DataFrame):
        if not hasattr(data, f'pivoted_{self.column}'):
            setattr(data, f'pivoted_{self.column}',
                    data.pivot(index='id', columns='d', values=self.column)
                    .reindex(data.id.unique()).values)

        return getattr(data, f'pivoted_{self.column}')


class AddChangeFeatures(PreprocessingStep):
    def __init__(self, shifts: List[int], column: str = "demand"):
        super().__init__()
        self.shifts = shifts
        self.column = column

    def process_merged(self, data: pd.DataFrame) -> DataType:
        for shift in self.shifts:
            shift_col_name = self._shift_col_name(shift)
            change_col_name = self._change_col_name(shift)

            intermediate_column_added = False  # Preserve column only if pre-existing
            if shift_col_name not in data.columns:
                intermediate_column_added = True
                data[shift_col_name] = data.groupby(["id"])[self.column].shift(shift)
            data[change_col_name] = (data[shift_col_name] - data[self.column]) / (
                data[shift_col_name]
            ).astype(np.float32)
            if intermediate_column_added:
                data.drop(shift_col_name, axis=1, inplace=True)

        return data

    def _shift_col_name(self, shift: int):
        return f'shift_{self.column}_t{shift}'

    def _change_col_name(self, shift: int):
        return f'{self.column}_change_t{shift}'

    def process_separate(self, data: DataTuple) -> DataType:
        return self._raise_invalid_type(data)


class AddRollingChangeFeatures(PreprocessingStep):
    def __init__(self, window_sizes: List[int], type: str, column: str = "demand"):
        super().__init__()
        self.window_sizes = window_sizes
        self.type = type
        self.column = column

    def process_merged(self, data: pd.DataFrame) -> DataType:
        for window in self.window_sizes:
            rolling_col_name = self._rolling_col_name(window)
            rolling_change_col_name = self._rolling_change_col_name(window)

            intermediate_column_added = False  # Preserve column only if pre-existing
            if rolling_col_name not in data.columns:
                intermediate_column_added = True
                data[rolling_col_name] = data.groupby(["id"])[self.column].transform(
                    lambda x: getattr(x.shift(1).rolling(window), self.type)()
                )
            data[rolling_change_col_name] = \
                (data[rolling_col_name] - data[self.column]) / (
                    data[rolling_col_name]
                ).astype(np.float32)

            if intermediate_column_added:
                data.drop(rolling_col_name, axis=1, inplace=True)
        return data

    def _rolling_col_name(self, window: int):
        return f"rolling_{self.column}_{self.type}_t{window}"

    def _rolling_change_col_name(self, window: int):
        return f"rolling_{self.column}_{self.type}_change_t{window}"

    def process_separate(self, data: DataTuple) -> DataType:
        return self._raise_invalid_type(data)


class AddTimeFeatures(PreprocessingStep):
    def __init__(self, time_attrs: List[str], cyclical: bool):
        super().__init__()
        self.time_attrs = time_attrs
        self.cyclical = cyclical

    def process_merged(self, data: pd.DataFrame) -> DataType:
        date = data.date.dt
        for attr in self.time_attrs:
            if self.cyclical:
                x = (getattr(date, attr) - (2000 if attr == 'year' else 0))
                x_min = x.min()
                x_max = x.max()
                data[f'{attr}_sin'] = np.sin(2 * np.pi * (x - x_min) / (x_max - x_min))
            else:
                data[attr] = (getattr(date, attr) - (2000 if attr == 'year' else 0)).astype(np.float32)

        return data

    def process_separate(self, data: DataTuple) -> DataType:
        return self._raise_invalid_type(data)


class AddWeekendFeature(PreprocessingStep):
    def __init__(self):
        super().__init__()

    def process_merged(self, data: pd.DataFrame) -> DataType:
        dayofweek = data['dayofweek'] if 'dayofweek' in data.columns else getattr(data['date'].dt, 'dayofweek')
        data["is_weekend"] = dayofweek.isin([5, 6]).astype(np.int8)
        return data

    def process_separate(self, data: DataTuple) -> DataType:
        return self._raise_invalid_type(data)


class RemoveNaNRows(PreprocessingStep):
    def __init__(self):
        super(RemoveNaNRows, self).__init__()

    def process_merged(self, data: pd.DataFrame) -> DataType:
        data.dropna(inplace=True)
        return data

    def process_separate(self, data: DataTuple) -> DataType:
        return self._raise_invalid_type(data)


class NaNsAsNumber(PreprocessingStep):
    def __init__(self, columns: List[str]):
        super(NaNsAsNumber, self).__init__()
        self.columns = columns

    def process_merged(self, data: pd.DataFrame) -> DataType:
        for col in self.columns:
            data[col].fillna(data[col].max() + 1, inplace=True)
        return data

    def process_separate(self, data: DataTuple) -> DataType:
        return self._raise_invalid_type(data)


class CategoricalToInt(PreprocessingStep):
    def __init__(self, categorical_columns: Optional[List[str]] = None,
                 categorical_columns_calendar: Optional[List[str]] = None,
                 categorical_columns_prices: Optional[List[str]] = None,
                 categorical_columns_sales: Optional[List[str]] = None):
        super(CategoricalToInt, self).__init__()
        self.categorical_columns = categorical_columns

        self.categorical_columns_calendar = categorical_columns_calendar
        self.categorical_columns_prices = categorical_columns_prices
        self.categorical_columns_sales = categorical_columns_sales

    def process_merged(self, data: pd.DataFrame) -> DataType:
        self._categorical_to_codes(data, self.categorical_columns)
        return data

    def process_separate(self, data: DataTuple) -> DataType:
        calendar, prices, sales = data
        self._categorical_to_codes(calendar, self.categorical_columns_calendar)
        self._categorical_to_codes(prices, self.categorical_columns_prices)
        self._categorical_to_codes(sales, self.categorical_columns_sales)
        return data

    @staticmethod
    def _categorical_to_codes(df: pd.DataFrame, cats: List[str]):
        for col in cats:
            df[col] = df[col].astype('category')
            df[col] = df[col].cat.codes
            df[col] -= df[col].min()
