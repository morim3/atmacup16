import time

import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm
import yaml

from tools import data_loader


def get_features(feature_list, df_yad, df_log, df_candidate, train_test="train"):
    '''
    df_yado: pl.DataFrame
        yad_no, yado_type, total_room_cnt, wireless_lan_flg, onsen_flg, kd_stn_5min, kd_bch_5min, kd_slp_5min, kd_conv_walk_5min
    df_log: pl.DataFrame
        session_id, seq_no, yad_no
    df_candidate: pl.DataFrame
        session_id, yad_no

    '''

    # yad_features = get_yado_features(yad_f, df_yad)
    # session_features = get_session_features(session_f, df_log)

    enumerated = df_log.with_columns(
        pl.arange(pl.count(), 0, -1).over("session_id").alias("enum")
    )
    pivoted = enumerated.pivot(
        index="session_id", columns="enum", values="yad_no")

    pivoted = pivoted.select(["session_id", "1", "2", "3", "4"])
    df_candidate = df_candidate.join(
        pivoted, how="left", left_on="session_id", right_on="session_id", )
    df_candidate = df_candidate.rename(
        {"1": "yad_no_1", "2": "yad_no_2", "3": "yad_no_3", "4": "yad_no_4", "yad_no": "yad_no_cand"})

    # feature
    for i in range(4):
        df_candidate = df_candidate.join(
            df_yad, how="left", left_on=f"yad_no_{i+1}", right_on="yad_no", suffix=(f'_{i+1}'))

    df_candidate = df_candidate.rename(
        {f"{col}": f"{col}_1" for col in df_yad.columns if col != "yad_no"})

    df_candidate = df_candidate.join(
        df_yad, how="left", left_on="yad_no_cand", right_on="yad_no", suffix="_cand")

    df_candidate = df_candidate.rename(
        {f"{col}": f"{col}_cand" for col in df_yad.columns if col != "yad_no"})

    # make matching features which means:
    # if {yad.columns}_{1,..,4} == {yad.columns}_cand then 1 else 0
    for i in range(4):
        for col in df_yad.columns:
            df_candidate = df_candidate.with_columns(
                (pl.col(f"{col}_{i+1}") == pl.col(f"{col}_cand")
                 ).cast(pl.Int8).alias(f"{col}_{i+1}_match")
            )

    # cat image features
    image_features = pl.read_parquet(
        "data/features/features_image_cluster.parquet")

    for i in range(4):
        df_candidate = df_candidate.join(
            image_features, how="left", left_on=f"yad_no_{i+1}", right_on="yad_no")
        df_candidate = df_candidate.rename(
            {f"{col}": f"{col}_{i+1}" for col in image_features.columns if col != "yad_no"})
        df_candidate = df_candidate.with_columns([
            pl.col(f"{col}_{i+1}").cast(pl.Int8) for col in image_features.columns if col != "yad_no"])

    df_candidate = df_candidate.join(
        image_features, how="left", left_on="yad_no_cand", right_on="yad_no", suffix="_cand")
    df_candidate = df_candidate.rename(
        {f"{col}": f"{col}_cand" for col in image_features.columns if col != "yad_no"})
    df_candidate = df_candidate.with_columns([
        pl.col(f"{col}_cand").cast(pl.Int8) for col in image_features.columns if col != "yad_no"])

    ## make image category matching features which means:
    ## if both {image_features.columns}_{1,2, 3, 4} and {image_features.columns}_cand are positive then 1 else 0
    for i in range(4):
        for col in image_features.columns:
            if col != "yad_no":
                df_candidate = df_candidate.with_columns(
                    (pl.col(f"{col}_{i+1}") - pl.col(f"{col}_cand")
                     ).cast(pl.Int8).alias(f"{col}_{i+1}_match")
                )

    # drop raw image features
    df_candidate = df_candidate.drop(
        [f"{col}_{i+1}" for col in image_features.columns for i in range(4) if col != "yad_no"])
    df_candidate = df_candidate.drop(
        [f"{col}_cand" for col in image_features.columns if col != "yad_no"])

    # make feature in discussion
    feature_name_list = [
        # 'latest_next_booking_top20',
        'past_view_yado',
        # 'top10_popular_yado',
        # 'top10_wid_popular_yado',
        'top10_ken_popular_yado',
        'top10_lrg_popular_yado',
        'top10_sml_popular_yado'
    ]

    for feature_name in tqdm(feature_name_list):
        print(feature_name)
        feature = pl.read_parquet(
            f'data/features/{train_test}_{feature_name}_feature.parquet')

        # TODO: nullの処理
        if train_test == 'train':
            # for fold in range(CFG.fold_num):
            if 'session_id' in feature.columns:
                df_candidate = df_candidate.join(feature, how='left', left_on=[
                                                 'session_id', 'yad_no_cand'], right_on=["session_id", "yad_no"])

            elif 'latest_yad_no' in feature.columns:
                feature = feature.with_columns(pl.col("fold").cast(pl.Int64), )
                df_candidate = df_candidate.join(feature, how='left', left_on=[
                                                 'fold', 'yad_no_1', 'yad_no_cand'], right_on=['fold', 'latest_yad_no', 'yad_no'])
            else:
                feature = feature.with_columns(pl.col("fold").cast(pl.Int64), )
                df_candidate = df_candidate.join(feature, how='left', left_on=[
                                                 'fold', 'yad_no_cand'], right_on=['fold', 'yad_no'])
        else:
            if 'session_id' in feature.columns:
                df_candidate = df_candidate.join(feature, how='left', left_on=[
                                                 'session_id', 'yad_no_cand'], right_on=["session_id", "yad_no"])
            elif 'latest_yad_no' in feature.columns:
                df_candidate = df_candidate.join(feature, how='left', left_on=[
                                                 'yad_no_1', 'yad_no_cand'], right_on=['latest_yad_no', 'yad_no'])
            else:
                df_candidate = df_candidate.join(feature, how='left', left_on=[
                                                 'yad_no_cand'], right_on=['yad_no'])

    df_candidate.with_columns([
        # pl.col("latest_next_booking_rank").cast(pl.Int8),
        pl.col("max_seq_no").cast(pl.Int8),
        pl.col("max_seq_no_diff").cast(pl.Int8),
        pl.col("session_view_count").cast(pl.Int8),
        pl.col("popular_ken_cd_rank").cast(pl.Int32),
        pl.col("popular_lrg_cd_rank").cast(pl.Int32),
        pl.col("popular_sml_cd_rank").cast(pl.Int32),
    ])

    df_candidate = df_candidate.with_columns([
        pl.col(f"total_room_cnt_{i+1}").cast(pl.Int16) for i in range(4)])

    # drop many category columns
    for i in range(4):
        for col in df_yad.columns:
            if col != "total_room_cnt":
                df_candidate = df_candidate.drop(f"{col}_{i+1}")

    for col in df_yad.columns:
        if col != "total_room_cnt" and col != "yad_no":
            df_candidate = df_candidate.drop(f"{col}_cand")

    df_candidate = df_candidate.drop(
        ["wid_cd", "ken_cd", "lrg_cd", "sml_cd", ])

    if train_test == "train":
        df_candidate = df_candidate.drop("fold")

    return df_candidate


def make_feature_and_label(candidate_train_tstamp, candidate_test_tstamp, featuer_list=None):

    loader = data_loader.AtmaData16Loader()
    train_log = loader.load_train_log()
    label_train = loader.load_cv_label()
    yado = loader.load_yado()
    test_log = loader.load_test_log()

    candidate = pl.read_parquet(
        f'data/candidates/train_candidate_{candidate_train_tstamp}.parquet')
    candidate_test = pl.read_parquet(
        f'data/candidates/test_candidate_{candidate_test_tstamp}.parquet')

    features_train = get_features(featuer_list, yado, train_log, candidate.join(
        label_train.select(["session_id", "fold"]), on=["session_id"]), train_test="train")
    features_test = get_features(
        featuer_list, yado, test_log, candidate_test, train_test="test")

    label_train = label_train.rename(
        {"yad_no": "yad_no_ans", })
    # if candidate["session_id", "yad_no"] is in train_label: then 1 else 0
    label_train = (
        candidate
        .select("session_id", "yad_no")
        .rename({"yad_no": "yad_no_cand"})
        .join(label_train, how="left", on="session_id")
        .select("session_id", "yad_no_cand", "yad_no_ans", "fold")
        .with_columns(
            pl.when(pl.col("yad_no_cand") == pl.col("yad_no_ans")).then(
                1).otherwise(0).alias("label")
        )
        .select(["session_id", "yad_no_cand", "fold", "label"])
        .sort("session_id", "yad_no_cand")
    )

    print(features_train.columns)
    features_train = features_train.lazy().sort(
        "session_id", "yad_no_cand").collect()
    features_test = features_test.lazy().sort(
        "session_id", "yad_no_cand").collect()

    timestamp = time.strftime("%Y%m%d%H%M%S")
    features_train.write_parquet(
        f'data/features/X_train_{candidate_train_tstamp}_{timestamp}.parquet')
    features_test.write_parquet(
        f'data/features/X_test_{candidate_test_tstamp}_{timestamp}.parquet')
    label_train.write_parquet(
        f'data/features/y_train_{candidate_train_tstamp}_{timestamp}.parquet')

    with open("data/features/latest.yaml", "w") as f:
        yaml.dump({
            "X_train": f"data/features/X_train_{candidate_train_tstamp}_{timestamp}.parquet",
            "X_test": f"data/features/X_test_{candidate_test_tstamp}_{timestamp}.parquet",
            "y_train": f"data/features/y_train_{candidate_train_tstamp}_{timestamp}.parquet",
        }, f)


if __name__ == "__main__":
    make_feature_and_label("20231213", "20231213")
