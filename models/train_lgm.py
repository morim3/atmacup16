import argparse
import gc
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm
import wandb
import yaml

from tools.etc import Timer, get_logger, seed_everything


def calc_apk(actual, predicted, k=10):
    if actual in predicted[:k]:
        return 1.0 / (predicted[:k].index(actual) + 1)
    return 0.0


def calc_mapk(actual, predicted, k=10):
    print(actual, predicted)
    return sum(calc_apk(a, list(p), k) for a, p in zip(actual, predicted)) / len(actual)


def specify_dtype(df, ):
    '''
    df: pd.DataFrame
        yad_no, yado_type, total_room_cnt_{"1", "2", "3", "4", "cand"}, wireless_lan_flg_{"1", "2", "3", "4", "cand"}, onsen_flg, kd_stn_5min, kd_bch_5min, kd_slp_5min, kd_conv_walk_5min, wid_cd, ken_cd, lrg_cd, sml_cd

    change alltype into int32
    '''

    for col in df.columns:
        df[col] = df[col].astype("category")

    for col in ["total_room_cnt"]:
        for i in ["", "_2", "_3", "_4"]:
            df[col+i] = df[col+i].astype("Int32")

    return df


def train_lgm(X, y, params, n_fold=5, seed=42, logger=None, ):
    '''
    X: pl.DataFrame
        session_id, seq_no, yad_no, yad_no_0, yad_no_1, ..., yad_no_9
    y: pl.DataFrame
        session_id, seq_no, yad_no, yad_no_0, yad_no_1, ..., yad_no_9
    params: dict
        lightgbm params
    n_splits: int
        number of folds
    seed: int
        random seed
    log: bool
        whether to log
    log_name: str
        wandb log name
    '''
    seed_everything(seed)
    # timer = Timer()
    # wandb init

    if logger is not None:
        # get wandb logger
        logger.log(params)

    # define cv
    oof = np.zeros(len(X))
    feature_importances = []
    for fold_id in range(n_fold):
        print("fold: ", fold_id)
        # get train data
        # valid index is the index of y["fold"] is same as fold_id
        y_index = y.with_columns(
            pl.arange(0, pl.count(), ).alias("index")
        )
        valid_index = y_index.filter(pl.col("fold") == fold_id)[
            "index"].to_list()
        X_train, X_valid = X[valid_index], X[valid_index]
        y_train, y_valid = y[valid_index], y[valid_index]

        X_train, X_valid = X_train.drop(["session_id", "seq_no", "yad_no", "1", "2", "3", "4"]).to_pandas(
        ), X_valid.drop(["session_id", "seq_no", "yad_no", "1", "2", "3", "4"]).to_pandas()
        y_train, y_valid = y_train.select(
            ["label"]).to_pandas(), y_valid.select(["label"]).to_pandas()

        X_train = specify_dtype(X_train)
        X_valid = specify_dtype(X_valid)

        # train model
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_valid, label=y_valid)

        model = lgb.train(
            params,
            train_data,
            valid_sets=[valid_data],
            num_boost_round=100,
            feval=None,
        )

        oof[valid_index] = model.predict(X_valid)

        # feature importance
        fold_importance = pd.DataFrame()
        fold_importance["feature"] = model.feature_name()
        fold_importance["importance"] = model.feature_importance()
        fold_importance["fold"] = fold_id
        feature_importances.append(fold_importance)

    oof = pd.DataFrame({
        'session_id': X['session_id'],
        'yad_no': X['yad_no'],
        'predict': oof
    })

    oof = oof.sort_values(['session_id', 'predict'], ascending=False)
    top_10 = create_top_10_yad_predict(oof)

    mapk = calc_mapk(actual=np.array(y.to_pandas().sort_values('session_id', ascending=True)['yad_no'].to_list()).astype(np.int32), predicted=np.array(top_10.values.tolist()).astype(np.int32), k=10)
    print("cv score: ", mapk)

    # log
    if logger is not None:
        logger.log_metrics({
            'cv': mapk,
        })
        time_stamp = time.strftime('%Y%m%d%H%M%S')
        # save feature importance
        feature_importances.to_csv('feature_importances.csv')
        logger.log_artifact('feature_importances.csv')
        # save model
        model.save_model(f'model_{time_stamp}.txt')
        logger.log_artifact(f'model_{time_stamp}.txt')

    return model, oof


def create_top_10_yad_predict(_df):

    _agg = _df.sort_values("predict", ascending=False).groupby(
        "session_id")["yad_no"].apply(list)

    out_df = pd.DataFrame(
        index=_agg.index, data=_agg.values.tolist()).iloc[:, :10]

    return out_df


def predict(X, model):
    '''
    X: pl.DataFrame
        session_id, seq_no, yad_no, yad_no_0, yad_no_1, ..., yad_no_9
    model_path: str
        path to model
    '''
    X = X.drop(["session_id", "seq_no", "yad_no",
               "1", "2", "3", "4"]).to_pandas()
    X = specify_dtype(X)
    return model.predict(X)


if __name__ == '__main__':
    # TODO: save model, save feature paths, parser
    # data_loader = AtmaData16Loader()
    # yado_df = data_loader.load_yado()
    # train_log_df = data_loader.load_train_log()
    # train_y = data_loader.load_cv_label()
    # test_log_df = data_loader.load_test_log()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_x_path", default="data/features/features_cand_train_20231211_20231212001534.parquet")
    parser.add_argument(
        "--train_y_path", default="data/features/label_cand_20231211_20231212001534.parquet")
    parser.add_argument(
        "--test_x_path", default="data/features/features_cand_test_20231211_20231212001534.parquet")
    parser.add_argument("--config_path", default="models/config_default.yml")
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--log_name", default="default")

    args = parser.parse_args()

    train_X = pl.read_parquet(args.train_x_path)
    train_y = pl.read_parquet(args.train_y_path)

    assert train_X["session_id"].equals(train_y["session_id"])

    with open(args.config_path) as f:
        config = yaml.safe_load(f)

    if args.log:
        logger = get_logger(args.log_name)
        logger.log_params(config)
        logger.log_params(args)

    else:
        logger = None

    # train
    model, oof = train_lgm(
        train_X, train_y, config["lgb_params"], n_fold=5, seed=42, logger=logger)

    del train_X, train_y
    gc.collect()
    # predict
    print("start prediction")
    test_X = pl.read_parquet(args.test_x_path)
    pred = predict(test_X, model)
    sub = create_top_10_yad_predict(pd.DataFrame({
        'session_id': test_X['session_id'],
        'yad_no': test_X['yad_no'],
        'predict': pred
    }))
    sub.columns = [f'predict_{c}' for c in sub.columns]
    sub = sub.reset_index(drop=True)
    # save
    pd.DataFrame(pred).to_csv('pred.csv', index=False)

    if logger is not None:
        logger.log_artifact('pred.csv')
