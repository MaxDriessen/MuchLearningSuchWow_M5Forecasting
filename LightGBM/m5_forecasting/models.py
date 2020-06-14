import os

import pandas as pd
import lightgbm as lgb

from typing import Dict, Any, Optional, List

import torch
from lightgbm.callback import CallbackEnv
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, TensorDataset

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Loss, RunningAverage, RootMeanSquaredError, MeanAbsoluteError

from ignite.contrib.handlers import ProgressBar

OptionalParams = Optional[Dict[str, Any]]


class Model(object):
    def __init__(self):
        pass

    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame,
              X_val: pd.DataFrame, y_val: pd.DataFrame):
        raise NotImplementedError()

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

    def save(self, path: str):
        raise NotImplementedError()


class LightGBMModel(Model):
    def __init__(self, model_file: Optional[str] = None, lgbm_params: OptionalParams = None,
                 lgbm_kwargs: OptionalParams = None, save_freq: int = -1):
        super().__init__()

        self.save_freq = save_freq
        if model_file is not None:
            self.model = lgb.Booster(model_file=model_file)
        else:
            self.model = None

        self.lgbm_params = {
            "objective": "poisson",
            "force_row_wise": True,
            "learning_rate": 0.075,
            "sub_row": 0.75,
            "bagging_freq": 1,
            "lambda_l2": 0.1,
            "metric": ["rmse"],
            'verbosity': 1,
            'num_iterations': 1600,
            'num_leaves': 128,
            "min_data_in_leaf": 100,
        }
        if lgbm_params is not None:
            for k, v in lgbm_params.items():
                self.lgbm_params[k] = v

        self.lgbm_kwargs = {
            "verbose_eval": 20
        }

        if lgbm_kwargs is not None:
            for k, v in lgbm_kwargs.items():
                self.lgbm_kwargs[k] = v

    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame,
              X_val: pd.DataFrame, y_val: pd.DataFrame):
        cat_cols = ['item_id', 'dept_id', 'store_id', 'cat_id', 'state_id'] + ["event_name_1", "event_name_2",
                                                                               "event_type_1", "event_type_2"]
        train_set = lgb.Dataset(
            X_train,
            label=y_train,
            categorical_feature=cat_cols,
        )

        val_set = lgb.Dataset(
            X_val,
            label=y_val,
            categorical_feature=cat_cols,
        )

        if not os.path.isdir('checkpoints'):
            os.mkdir('checkpoints')

        def save_checkpoints(env: CallbackEnv):
            if self.save_freq > 0 and env.iteration % self.save_freq == 0:
                env.model.save_model(os.path.join(
                    'checkpoints',
                    f'lgb_model_{env.iteration}_{env.evaluation_result_list[1][2]}.txt'
                )
                )

        self.model = lgb.train(
            self.lgbm_params,
            train_set,
            valid_sets=[train_set, val_set],
            valid_names=["train", "valid"],
            callbacks=[save_checkpoints],
            **self.lgbm_kwargs,
        )

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return self.model.predict(X)

    def save(self, path: str):
        if self.model is None:
            raise ValueError('Must train model before saving')

        self.model.save_model(path)


class NeuralRegressionModel(Model):
    class MLP(nn.Module):
        def __init__(self, n_input: int, n_hidden: List[int]):
            super().__init__()
            self.fcs = nn.ModuleList()
            self.bns = nn.ModuleList()
            self.dos = nn.ModuleList()
            n_prev = n_input
            for h in n_hidden:
                self.fcs.append(nn.Linear(n_prev, h))
                self.bns.append(nn.BatchNorm1d(h))
                self.dos.append(nn.Dropout())
                n_prev = h

            self.fc_output = nn.Linear(n_prev, 1)

        def forward(self, x):
            for fc, bn, do in zip(self.fcs, self.bns, self.dos):
                x = F.relu(bn(fc(x)))
                x = do(x)

            x = F.relu(self.fc_output(x))
            return x

    def __init__(self, model_file: Optional[str] = None, n_hidden: Optional[List[int]] = None,
                 device: Optional[torch.device] = None):
        super().__init__()

        self.device = device if device is not None else torch.device('cpu')
        self.model = None
        self.optim = None
        self.loss = nn.MSELoss()

        self.n_hidden = n_hidden if n_hidden is not None else [32, 64]
        self.model_file = model_file

    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame,
              X_val: pd.DataFrame, y_val: pd.DataFrame):
        assert len(X_train.columns) == len(X_val.columns)
        X_train = pd.get_dummies(X_train, columns=['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2'])
        X_train.fillna(0, inplace=True)
        X_train.drop(columns=['item_id'], inplace=True)
        X_val = pd.get_dummies(X_val, columns=['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2'])
        X_val.fillna(0, inplace=True)
        X_val.drop(columns=['item_id'], inplace=True)
        X_val = X_val.T.reindex(list(set(X_train.columns))).T.fillna(0)
        X_val = X_val[X_train.columns]

        assert (X_train.columns == X_val.columns).all()

        self.model = NeuralRegressionModel.MLP(len(X_train.columns), self.n_hidden)
        if self.model_file is not None:
            self.model.load_state_dict(torch.load(self.model_file))

        X_train = torch.tensor(X_train.values).float()
        y_train = torch.tensor(y_train.values).float()
        X_val = torch.tensor(X_val.values).float()
        y_val = torch.tensor(y_val.values).float()
        assert not (bool(torch.isnan(X_train).any()) or bool(torch.isnan(y_train).any()) or
                    bool(torch.isnan(X_val).any()) or bool(torch.isnan(y_val).any()))

        self.optim = Adam(self.model.parameters())

        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(dataset=train_dataset, batch_size=2048, shuffle=True)
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(dataset=val_dataset, batch_size=2048, shuffle=True)

        trainer = create_supervised_trainer(self.model, self.optim, self.loss, device=self.device)
        evaluator = create_supervised_evaluator(
            self.model, metrics={"rmse": RootMeanSquaredError(), "mae": MeanAbsoluteError()}, device=self.device
        )

        RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")

        pbar = ProgressBar(persist=True)
        pbar.attach(trainer, metric_names="all")

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(engine):
            evaluator.run(train_loader)
            metrics = evaluator.state.metrics
            avg_rmse = metrics["rmse"]
            avg_mae = metrics["mae"]
            pbar.log_message(
                "Training Results - Epoch: {}  Avg rmse: -{:.2f} Avg mae: {:.2f}".format(
                    engine.state.epoch, avg_rmse, avg_mae
                )
            )

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            evaluator.run(val_loader)
            metrics = evaluator.state.metrics
            avg_rmse = metrics["rmse"]
            avg_mae = metrics["mae"]
            pbar.log_message(
                "Validation Results - Epoch: {}  Avg rmse: {:.2f} Avg mae: {:.2f}".format(
                    engine.state.epoch, avg_rmse, avg_mae
                )
            )

            pbar.n = pbar.last_print_n = 0

        trainer.run(train_loader, max_epochs=20)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        X.drop(columns=['item_id'], inplace=True)
        assert len(X.columns) == 77
        X.fillna(0, inplace=True)

        self.model = NeuralRegressionModel.MLP(len(X.columns), self.n_hidden)
        if self.model_file is not None:
            self.model.load_state_dict(torch.load(self.model_file))
            self.model.eval()
        with torch.no_grad():
            return self.model(torch.tensor(X.values).float()).numpy()

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)
