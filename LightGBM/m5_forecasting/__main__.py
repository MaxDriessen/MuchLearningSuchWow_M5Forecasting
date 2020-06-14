import argparse
import gc
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any

import pandas as pd
import tables

from m5_forecasting.loader import DataTuple, load_data_raw
from m5_forecasting.preprocessing import ToLongFormat, MergeData, \
    AddTimeFeatures, ToDask, DownCastTypes, AddRollFeatures, AddShiftFeatures, \
    AddRollingChangeFeatures, AddWeekendFeature, ExtendData, TemporallyCropData, AddChangeFeatures, \
    RemoveNaNRows, NaNsAsNumber, CategoricalToInt, PreprocessingStep, DropSeries
from m5_forecasting.models import NeuralRegressionModel, Model
from m5_forecasting.utils import prepare_regression_data, train_val_split_random, reset_dataframe_pivot_cache

logger = logging.getLogger(__name__)

TRAIN_DATA_MAX = 1913
VAL_DATA_MAX = TRAIN_DATA_MAX + 28
EVAL_DATA_MAX = TRAIN_DATA_MAX + 28 * 2

PREPROCESSING_OPTIONS = {
    'extend_data': ExtendData,
    'temporal_crop': TemporallyCropData,
    'to_long': ToLongFormat,
    'merge_data': MergeData,
    'downcast': DownCastTypes,
    'to_dask': ToDask,
    'roll': AddRollFeatures,
    'shift': AddShiftFeatures,
    "shift_change": AddChangeFeatures,
    "rolling_change": AddRollingChangeFeatures,
    "time_features": AddTimeFeatures,
    "weekend_feature": AddWeekendFeature,
    'remove_nan_rows': RemoveNaNRows,
    'nans_as_number': NaNsAsNumber,
    "categorical_to_int": CategoricalToInt,
    'drop_series': DropSeries
}


def construct_steps(file):
    cfg = json.load(file)
    if len(cfg) == 0:
        return [], []

    steps = [(PREPROCESSING_OPTIONS[step['type']](**step.get('params', {})), step.get('params', {})) for step in cfg]
    return zip(*steps)


def preprocess_with_steps(data, steps, params):
    from tqdm import tqdm
    for step, step_params in tqdm(zip(steps, params), total=(len(steps))):
        tqdm.write(step.__class__.__name__)
        tqdm.write(str(step_params))

        data = step.process(data)
        if type(data) is pd.DataFrame:
            tqdm.write(str(data.dtypes))
        else:
            tqdm.write(str(data.sales.dtypes))
        gc.collect()

    return data


def preprocess(args):
    data = load_data_raw(read_csvs=args.read_csvs or args.evaluation, evaluation=args.evaluation)
    steps, params = construct_steps(args.config)

    data = preprocess_with_steps(data, steps, params)

    if type(data) is DataTuple:
        data.calendar.to_hdf(args.output, key='calendar')
        data.prices.to_hdf(args.output, key='prices')
        data.sales.to_hdf(args.output, key='sales')
    elif isinstance(data, pd.DataFrame):
        try:
            data.to_hdf(args.output, key='data')
        except tables.exceptions.HDF5ExtError:
            data.to_feather(args.output, key='data')
    else:
        raise ValueError(f'Invalid data type: {type(data)}')


def data_to_hdf5(_):
    from loader import load_data
    print('Loading data...')
    calendar, prices, sales = load_data()

    print('Saving as hdf5...')
    calendar.to_hdf('data/data.h5', key='calendar')
    prices.to_hdf('data/data.h5', key='prices')
    sales.to_hdf('data/data.h5', key='sales')


def train(args):
    from models import LightGBMModel

    print("Loading data...")
    data = pd.read_hdf(args.input.name, key='data')
    assert isinstance(data, pd.DataFrame)

    print("Splitting data...")
    X, y = prepare_regression_data(data, max_d=VAL_DATA_MAX if args.evaluation else TRAIN_DATA_MAX)
    del data

    X_train, y_train, X_val, y_val = train_val_split_random(X, y)
    del X, y

    if args.model == 'lightgbm':
        model = LightGBMModel(save_freq=200, lgbm_params={'bagging_fraction': 0.9038241980952565,
                                                          'bagging_freq': 2,
                                                          'colsample_bytree': 0.35538420604932075,
                                                          'lambda_l2': 0.06,
                                                          'max_bin': 511,
                                                          'min_data_in_leaf': 71,
                                                          'num_leaves': 360})
    else:
        import torch
        model = NeuralRegressionModel(device=torch.device('cuda' if args.gpu else 'cpu'))

    print("Training model...")
    model.train(X_train, y_train, X_val, y_val)

    print("Saving model...")
    model.save(args.output.name)


def predict(args):
    from models import LightGBMModel

    print("Loading data...")

    data = load_data_raw(read_csvs=args.evaluation, evaluation=args.evaluation)
    steps, params = construct_steps(args.config)

    data = ExtendData(upto=EVAL_DATA_MAX if args.evaluation else VAL_DATA_MAX)(data)
    data = CategoricalToInt(
        categorical_columns_calendar=["event_name_1", "event_type_1", "event_name_2",
                                      "event_type_2", "weekday"],
        categorical_columns_prices=["store_id", "item_id"],
        categorical_columns_sales=["item_id", "dept_id", "store_id", "cat_id", "state_id"])(data)
    data = ToLongFormat()(data)
    data = MergeData()(data)
    data = DownCastTypes()(data)

    print("Splitting data...")

    if args.model == 'lightgbm':
        model = LightGBMModel(args.model_file.name)
    else:
        import torch
        model = NeuralRegressionModel(model_file=args.model_file.name,
                                      device=torch.device('cuda' if args.gpu else 'cpu'))

    if args.magic:
        sub = magic_predict(data, steps, params, model, args.evaluation)
    else:
        sub = regular_predict(data, steps, params, model, args.evaluation)

    sub.to_csv(args.output.name, index=False)
    print((sub.iloc[:, 1:] > 1).sum())


def regular_predict(data: pd.DataFrame, steps: List[PreprocessingStep],
                    params: List[Dict[str, Any]], model: Model, evaluation: bool = False) -> pd.DataFrame:
    fday = datetime(2016, 4, 25)

    useless_cols = ["id", "date", "demand", "d", "wm_yr_wk"]

    max_lags = 57

    data.date = pd.to_datetime(data.date)

    pred_range = range(28, 56) if evaluation else range(0, 28)

    for tdelta in pred_range:
        reset_dataframe_pivot_cache(data)
        _process_and_predict(data, model, steps, params, fday, tdelta, max_lags, useless_cols)

    sub = _to_submission_format(data)

    return sub


def magic_predict(data: pd.DataFrame, steps: List[PreprocessingStep],
                  params: List[Dict[str, Any]], model: Model, evaluation: bool = False) -> pd.DataFrame:
    alphas = [1.028, 1.023, 1.018]
    weights = [1 / len(alphas)] * len(alphas)
    sub = 0.

    fday = datetime(2016, 4, 25)

    useless_cols = ["id", "date", "demand", "d", "wm_yr_wk"]

    max_lags = 57
    cols = [f"F{i}" for i in range(1, 29)]

    data.date = pd.to_datetime(data.date)

    pred_range = range(28, 56) if evaluation else range(0, 28)

    for icount, (alpha, weight) in enumerate(zip(alphas, weights)):
        for tdelta in pred_range:
            reset_dataframe_pivot_cache(data)
            _process_and_predict(data, model, steps, params, fday, tdelta, max_lags, useless_cols,
                                 alpha=alpha)

        te_sub = _to_submission_format(data)

        if icount == 0:
            sub = te_sub
            sub[cols] *= weight
        else:
            sub[cols] += te_sub[cols] * weight
        print(icount, alpha, weight)

    return sub


def _process_and_predict(data: pd.DataFrame, model: Model, steps: List[PreprocessingStep],
                         params: List[Dict[str, Any]], fday: datetime, tdelta: int,
                         max_lags: int, useless_cols: List[str], alpha: float = 1.0):
    day = fday + timedelta(days=tdelta)
    print(tdelta, day)
    tst = data[(data.date >= day - timedelta(days=max_lags)) & (data.date <= day)].copy()
    tst = preprocess_with_steps(tst, steps, params)
    input_cols = tst.columns[~tst.columns.isin(useless_cols)]
    tst = tst.loc[tst.date == day, input_cols]
    data.loc[data.date == day, "demand"] = alpha * model.predict(tst)  # magic multiplier by kyakovlev


def _to_submission_format(data: pd.DataFrame) -> pd.DataFrame:
    n = data.d.max() - TRAIN_DATA_MAX
    assert data.d.min() <= TRAIN_DATA_MAX + 1 and (n == 28 or n == 56)
    data.to_csv('test.csv', index=False)

    cols = [f"F{i}" for i in range(1, 29)]
    sub = data.loc[data.d >= TRAIN_DATA_MAX + 1, ["id", "demand", "d"]].copy()

    sub.loc[sub.d > VAL_DATA_MAX, 'id'] = sub[sub.d > VAL_DATA_MAX].id.str.replace('validation', 'evaluation')
    sub.loc[sub.d <= VAL_DATA_MAX, 'id'] = sub[sub.d <= VAL_DATA_MAX].id.str.replace('evaluation', 'validation')

    sub.loc[sub.d <= VAL_DATA_MAX, 'F'] = [f"F{rank - TRAIN_DATA_MAX}" for rank in sub.loc[sub.d <= VAL_DATA_MAX, 'd']]
    sub.loc[sub.d > VAL_DATA_MAX, 'F'] = [f"F{rank - VAL_DATA_MAX}" for rank in sub.loc[sub.d > VAL_DATA_MAX, 'd']]
    sub = sub.set_index(["id", "F"]).unstack()["demand"][cols].reset_index()
    sub.fillna(0., inplace=True)
    sub.sort_values("id", inplace=True)
    sub.reset_index(drop=True, inplace=True)
    if n == 28:
        # Duplicate validation predictions
        sub2 = sub.copy()
        sub2["id"] = sub2["id"].str.replace("validation$", "evaluation")
        sub = pd.concat([sub, sub2], axis=0, sort=False)
    return sub


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser.add_argument('--debug', default=False, action='store_true')

    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('input', type=argparse.FileType('r'))
    train_parser.add_argument('-model', type=str, required=True, choices=['lightgbm', 'nnregr'])
    train_parser.add_argument('-output', type=argparse.FileType('w'), required=True)
    train_parser.add_argument('--evaluation', default=False, action='store_true')
    train_parser.add_argument('--gpu', default=False, action='store_true')
    train_parser.set_defaults(func=train)

    predict_parser = subparsers.add_parser('predict')
    predict_parser.add_argument('config', type=argparse.FileType('r'))
    predict_parser.add_argument('model', type=str, choices=['lightgbm', 'nnregr'])
    predict_parser.add_argument('model_file', type=argparse.FileType('r'))
    predict_parser.add_argument('output', type=argparse.FileType('w'))
    predict_parser.add_argument('--gpu', default=False, action='store_true')
    predict_parser.add_argument('--magic', default=False, action='store_true')
    predict_parser.add_argument('--evaluation', default=False, action='store_true')
    predict_parser.set_defaults(func=predict)

    to_hdf5 = subparsers.add_parser('data-to-hdf5')
    to_hdf5.set_defaults(func=data_to_hdf5)

    preprocess_parser = subparsers.add_parser('preprocess')
    preprocess_parser.add_argument('--output', type=str, required=True)
    preprocess_parser.add_argument('config', type=argparse.FileType('r'))
    preprocess_parser.add_argument('--read-csvs', dest='read_csvs', default=False, action='store_true')
    preprocess_parser.add_argument('--evaluation', default=False, action='store_true')

    preprocess_parser.set_defaults(func=preprocess)

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format='%(message)s')

    args.func(args)


if __name__ == '__main__':
    main()
